import torch
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)


class TRTModule:
    def __init__(self, engine_path) -> None:
        super().__init__()

        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        self.shapes = {}
        self.dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            self.shapes[name] = tuple(self.engine.get_tensor_shape(name))

            self.dtypes[name] = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
            }[self.engine.get_tensor_dtype(name)]

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.outputs = [
            torch.empty(self.shapes[n], dtype=self.dtypes[n], device="cuda")
            for n in self.output_names
        ]
        for name, out in zip(self.output_names, self.outputs):
            self.context.set_tensor_address(name, out.data_ptr())

    def __call__(self, *inputs):
        assert len(inputs) == len(self.input_names)
        for name, t in zip(self.input_names, inputs):
            assert (
                t.is_cuda and t.is_contiguous() and t.dtype == self.dtypes[name]
            ), f"{name}: expected {self.dtypes[name]} contiguous cuda, got {t.dtype}"
            self.context.set_tensor_address(name, t.data_ptr())

        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)

        return self.outputs[0] if len(self.outputs) == 1 else tuple(self.outputs)
