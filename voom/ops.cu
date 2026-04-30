#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template<typename T>
__global__ void lift_splat_gather_kernel(
    const torch::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> context,
    const torch::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> depth,
    const torch::PackedTensorAccessor32<long, 1, at::RestrictPtrTraits> offsets,
    const torch::PackedTensorAccessor32<long, 1, at::RestrictPtrTraits> pixs,
    const int B,
    const int CH,
    const int N_VOXELS,
    const int GX,
    const int GY,
    const int GZ,
    torch::PackedTensorAccessor32<T, 5, at::RestrictPtrTraits> grid_out
){
    const int vox = blockIdx.x * blockDim.x + threadIdx.x;
    const int ch = blockIdx.y * blockDim.y + threadIdx.y;

    if ((vox >= N_VOXELS) || (ch >= CH)){
        return;
    }

    const int gx = vox / (GY * GZ);
    const int gy = (vox / GZ) % GY;
    const int gz = vox % GZ;

    const int start = offsets[vox];
    const int end = offsets[vox + 1];

    for (int b = 0; b < B; b++){
        T sum{};

        for (int i = start; i < end; i++){
            const long packed = pixs[i];

            const int di = packed >> 32;
            const int py = (packed >> 16) & 0xFFFF;
            const int px = packed & 0xFFFF;

            const T c = context[b][ch][py][px];
            const T d = depth[b][di][py][px];

            sum += c * d;
        }

        grid_out[b][ch][gx][gy][gz] = sum;
    }
}

torch::Tensor lift_splat_gather(
    torch::Tensor context,
    torch::Tensor depth,
    torch::Tensor offsets,
    torch::Tensor pixs,
    int GX,
    int GY,
    int GZ
) {
    const int B = context.size(0);
    const int CH = context.size(1);
    const int N_VOXELS = GX * GY * GZ;

    torch::Tensor voxels = torch::zeros({B, CH, GX, GY, GZ}, context.options());

    dim3 Block(16, 16);
    dim3 Grid(
        (N_VOXELS + 15) / 16,
        (CH + 15) / 16
    );

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(depth.scalar_type(), "lift-splat kernel error", [&] {
            lift_splat_gather_kernel<<<Grid, Block, 0, stream>>>(
                context.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                depth.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                offsets.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
                pixs.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
                B, CH, N_VOXELS, GX, GY, GZ,
                voxels.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>()
            );
        }
    );

    return voxels;
}


template<int CH>
__global__ void lift_splat_gather_fp16_nhwc_ch64_kernel(
    const torch::PackedTensorAccessor32<at::Half, 4, at::RestrictPtrTraits> context,
    const torch::PackedTensorAccessor32<at::Half, 4, at::RestrictPtrTraits> depth,
    const torch::PackedTensorAccessor32<long, 1, at::RestrictPtrTraits> offsets,
    const torch::PackedTensorAccessor32<long, 1, at::RestrictPtrTraits> pixs,
    const int B,
    const int GX,
    const int GY,
    const int GZ,
    torch::PackedTensorAccessor32<at::Half, 5, at::RestrictPtrTraits> grid_out
){
    const int N_VOXELS = GX * GY * GZ;
    const int vox = blockIdx.x * blockDim.x + threadIdx.x;

    if (vox >= N_VOXELS) {
        return;
    }

    const int gx = vox / (GY * GZ);
    const int gy = (vox / GZ) % GY;
    const int gz = vox % GZ;

    const int start = offsets[vox];
    const int end = offsets[vox + 1];

    float sum[CH];
    #pragma unroll
    for (int c = 0; c < CH; c++) {
        sum[c] = 0.f;
    }

    for (int b = 0; b < B; b++) {
        for (int i = start; i < end; i++) {
            const long packed = pixs[i];
            const int di = (int)(packed >> 32);
            const int py = (int)((packed >> 16) & 0xFFFF);
            const int px = (int)(packed & 0xFFFF);

            const float d = __half2float(*reinterpret_cast<const __half*>(&depth[b][di][py][px]));
            const __half* cvec = reinterpret_cast<const __half*>(&context[b][py][px][0]);

            #pragma unroll
            for (int c = 0; c < CH; c += 8) {
                uint4 v = *reinterpret_cast<const uint4*>(cvec + c);
                __half2 h01 = *reinterpret_cast<__half2*>(&v.x);
                __half2 h23 = *reinterpret_cast<__half2*>(&v.y);
                __half2 h45 = *reinterpret_cast<__half2*>(&v.z);
                __half2 h67 = *reinterpret_cast<__half2*>(&v.w);
                sum[c+0] += __half2float(__low2half (h01)) * d;
                sum[c+1] += __half2float(__high2half(h01)) * d;
                sum[c+2] += __half2float(__low2half (h23)) * d;
                sum[c+3] += __half2float(__high2half(h23)) * d;
                sum[c+4] += __half2float(__low2half (h45)) * d;
                sum[c+5] += __half2float(__high2half(h45)) * d;
                sum[c+6] += __half2float(__low2half (h67)) * d;
                sum[c+7] += __half2float(__high2half(h67)) * d;
            }
        }

        #pragma unroll
        for (int c = 0; c < CH; c++) {
            grid_out[b][c][gx][gy][gz] = __float2half(sum[c]);
        }
    }
}


torch::Tensor lift_splat_gather_fp16_nhwc_ch64(
    torch::Tensor context,
    torch::Tensor depth,
    torch::Tensor offsets,
    torch::Tensor pixs,
    int GX,
    int GY,
    int GZ
) {
    TORCH_CHECK(context.scalar_type() == at::kHalf, "context must be fp16");
    TORCH_CHECK(depth.scalar_type()   == at::kHalf, "depth must be fp16");
    TORCH_CHECK(context.is_contiguous(), "context must be contiguous (NHWC)");
    TORCH_CHECK(depth.is_contiguous(),   "depth must be contiguous");

    const int B  = context.size(0);
    const int CH = context.size(3);
    const int N_VOXELS = GX * GY * GZ;

    TORCH_CHECK(CH == 64, "kernel built for CH=64");

    auto opts = context.options();
    torch::Tensor voxels = torch::empty({B, CH, GX, GY, GZ}, opts);

    const int threads = 128;
    const int blocks  = (N_VOXELS + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream();
    lift_splat_gather_fp16_nhwc_ch64_kernel<64><<<blocks, threads, 0, stream>>>(
        context.packed_accessor32<at::Half, 4, at::RestrictPtrTraits>(),
        depth.packed_accessor32<at::Half, 4, at::RestrictPtrTraits>(),
        offsets.packed_accessor32<long, 1, at::RestrictPtrTraits>(),
        pixs.packed_accessor32<long, 1, at::RestrictPtrTraits>(),
        B, GX, GY, GZ,
        voxels.packed_accessor32<at::Half, 5, at::RestrictPtrTraits>()
    );

    return voxels;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lift_splat_gather", &lift_splat_gather, "lift-splat fused cuda kernel");
    m.def("lift_splat_gather_fp16_nhwc_ch64", &lift_splat_gather_fp16_nhwc_ch64, "lift-splat fp16 NHWC ch=64 vectorized");
}
