# rev7 — deployed checkpoint

Weights are not in this folder, they're hosted on HF:

```bash
./scripts/download_weights.sh   # release/voom.safetensors
```

| Metric                 | Value / Details                                  |
|------------------------|--------------------------------------------------|
| Occ IoU                | 0.3159                                           |
| Occ P                  | 0.4399                                           |
| Occ R                  | 0.5285                                           |
| Occ F1                 | 0.4802                                           |
| Sem Accuracy           | 0.7981                                           |

**Per-class IoU**

| Class           | IoU     |
|-----------------|---------|
| car             | 0.2036  |
| bicycle         | 0.0000  |
| motorcycle      | 0.0077  |
| truck           | 0.1186  |
| other-vehicle   | 0.0371  |
| person          | 0.0000  |
| bicyclist       | 0.0002  |
| motorcyclist    | 0.0000  |
| road            | 0.4381  |
| parking         | 0.1427  |
| sidewalk        | 0.2247  |
| other-ground    | 0.0002  |
| building        | 0.0566  |
| fence           | 0.0640  |
| trunk           | 0.1980  |
| terrain         | 0.0055  |
| pole            | 0.0062  |
| traffic-sign    | 0.0002  |

**mIoU: 0.0835**
