# MCBASegFormer

## 📥 Dataset Download & Deployment Guide

Please download the standardized benchmarks from the cloud storage links below and place them directly into the `./dataset/` directory.

### 1. Training Dataset
* **Download Link:** [[Google Drive Link (399.5MB)]](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view)
* **Description:** Contains a total of 1450 clinical frames spanning two sub-datasets: **Kvasir-SEG** (900 train samples) and **CVC-ClinicDB** (550 train samples).
* **Deployment Execution:** Unzip the archive and ensure the contents are mapped to `./dataset/TrainDataset/`.

### 2. Testing Dataset
* **Download Link:** [Google Drive Link (327.2MB)]
* **Description:** Contains five standard verification benchmarks: **CVC-300** (60 test samples), **CVC-ClinicDB** (62 test samples), **CVC-ColonDB** (380 test samples), **ETIS-LaribPolypDB** (196 test samples), and **Kvasir** (100 test samples).
* **Deployment Execution:** Unzip the archive and ensure the contents are mapped to `./dataset/TestDataset/`.

---

## 📂 Dataset Directory Structure & Dataset Setup
```text
dataset/
├── TrainDataset/               # 联合训练数据集
│   ├── CVC-ClinicDB/           # 550 个训练样本
│   │   ├── images/
│   │   └── masks/
│   └── Kvasir-SEG/             # 900 个训练样本
│       ├── images/
│       └── masks/
└── TestDataset/                # 独立测试验证数据集 
    ├── CVC-300/                # 60 个测试样本
    │   ├── images/
    │   └── masks/
    ├── CVC-ClinicDB/           # 62 个测试样本
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/            # 380 个测试样本
    │   ├── images/
    │   └── masks/
    ├── ETIS-LaribPolypDB/      # 196 个测试样本
    │   ├── images/
    │   └── masks/
    └── Kvasir/                 # 100 个测试样本
        ├── images/
        └── masks/

```
