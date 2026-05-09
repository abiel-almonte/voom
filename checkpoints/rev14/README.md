# rev14 — best checkpoint

Weights are not in this folder, they're hosted on HF:

```bash
./scripts/download_weights.sh   # release/voom-rev14.safetensors
```

| Metric                 | Value / Details                                  |
|------------------------|--------------------------------------------------|
| Occ IoU                | 0.3143                                           |
| Occ P                  | 0.4182                                           |
| Occ R                  | 0.5584                                           |
| Occ F1                 | 0.4783                                           |
| Sem Accuracy           | 0.4561                                           |

**Per-class IoU**

| Class           | IoU     |
|-----------------|---------|
| car             | 0.1894  |
| bicycle         | 0.0072  |
| motorcycle      | 0.0382  |
| truck           | 0.0755  |
| other-vehicle   | 0.0322  |
| person          | 0.0004  |
| bicyclist       | 0.0059  |
| motorcyclist    | 0.0000  |
| road            | 0.4303  |
| parking         | 0.1320  |
| sidewalk        | 0.2206  |
| other-ground    | 0.0000  |
| building        | 0.0273  |
| fence           | 0.0677  |
| vegetation      | 0.1754  |
| trunk           | 0.0091  |
| terrain         | 0.2041  |
| pole            | 0.0151  |
| traffic-sign    | 0.0031  |

**mIoU: 0.0860**
