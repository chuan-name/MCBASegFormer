# MCBASegFormer
## 📂 Project Directory Structure & Dataset Setup
```text
dataset/
├── TrainDataset/               # 联合训练数据集 (399.5MB)
│   ├── CVC-ClinicDB/           # 550 个训练样本
│   │   ├── images/
│   │   └── masks/
│   └── Kvasir-SEG/             # 900 个训练样本
│       ├── images/
│       └── masks/
└── TestDataset/                # 独立测试验证数据集 (327.2MB)
    ├── CVC-300/                # 60 个测试样本
    │   ├── images/
    │   └── masks/
    ├── CVC-ClinicDB/           # 62 个测试样本
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/            # 380 个测试样本
    │   ├── images/
    │   └── masks/
    ├── ETIS-LaribPolypDB/      # 196 个测试样本 (核心突破泛化靶向集)
    │   ├── images/
    │   └── masks/
    └── Kvasir/                 # 100 个测试样本
        ├── images/
        └── masks/

```
