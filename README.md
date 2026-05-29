# MCBASegFormer

## 🛠️ Environment Setup & Installation

To eliminate dependency conflicts and ensure smooth reproduction on an NVIDIA RTX 3060 GPU, we recommend managing your runtime framework via Anaconda. Execute the following commands sequentially to initialize the self-contained environment:

### 1. Create and activate the isolated conda sub-environment
conda create -n mcbaseg python=3.9.13 -y

conda activate mcbaseg

### 2. Install PyTorch ecosystem configured with CUDA 12.1 (Production Release)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

## 📥 Dataset Download & Deployment Guide

Please download the standardized benchmarks from the cloud storage links below and place them directly into the `./dataset/` directory.

### 1. Training Dataset
* **Download Link:** [[Google Drive Link (399.5MB)]](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view)
* **Description:** Contains a total of 1450 clinical frames spanning two sub-datasets: **Kvasir-SEG** (900 train samples) and **CVC-ClinicDB** (550 train samples).
* **Deployment Execution:** Unzip the archive and ensure the contents are mapped to `./dataset/TrainDataset/`.

### 2. Testing Dataset
* **Download Link:** [[Google Drive Link (327.2MB)]](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view)
* **Description:** Contains five standard verification benchmarks: **CVC-300** (60 test samples), **CVC-ClinicDB** (62 test samples), **CVC-ColonDB** (380 test samples), **ETIS-LaribPolypDB** (196 test samples), and **Kvasir** (100 test samples).
* **Deployment Execution:** Unzip the archive and ensure the contents are mapped to `./dataset/TestDataset/`.

---

## 📂 Dataset Directory Topology (Final Verification)

After extraction, your local repository must mirror the exact structured path below before running `Train.py` or `Test.py`:

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
## 🚀 How to Run Training

The training hyperparameters and dataset directory pathways are managed natively via Python's `argparse` control layer inside `Train.py`. 

### 1. Default Directory Matching
By default, the script looks for the training partition inside `./dataset/TrainDataset/`. Please ensure your data folders are organized correctly according to the topology chart above before launching.

### 2. Single-GPU Standard Training
To launch the multi-scale deep supervision optimization flow using our default runtime setup (e.g., Batch Size of 8, Learning Rate of 0.0003, and 80 Epochs) on a single NVIDIA RTX 3060 GPU, execute the entry script directly from your terminal:

```bash
python Train.py
```

### 3. Custom Parameter Adjustments (Via Command Line)
If you wish to override the default hardcoded parameters (such as changing the batch size to fit lower VRAM limits, or altering the training epoch limits) without modifying the Python source code, append the respective argument flags as shown below:

```bash
python Train.py --batch_size 4 --epochs 100 --lr 0.0005
```

## 🧪 How to Run Testing (Inference & Saliency Map Generation)

The quantitative testing pipeline is driven natively via **`Test.py`**. Unlike traditional sequential testing frameworks, our script is completely automated. It will automatically loop through all five benchmark sub-datasets stored inside `./dataset/TestDataset/` and export the fine-grained, edge-refined prediction masks in one single execution block.

### 1. Default Verification of Network Checkpoint
By default, the framework searches for your optimized model weights inside the following relative path:
`./checkpoints/MCBASegFormer/MCBASegFormer.pth`

Please ensure that your trained weight file is named and positioned correctly before launching the evaluation segment.

### 2. Execution of Unified Inference Run
To generate the binary saliency maps across the entire evaluation horizon (including **Kvasir**, **CVC-ClinicDB**, and the zero-shot unseen generalization benchmarks like **ETIS-LaribPolypDB**), run the script directly from your terminal with zero parameter workload:

```bash
python Test.py
```

### 3. Alternative Weight Overriding (Optional Command Argument)
If you wish to test a specific checkpoint saved in an alternative folder without modifying the underlying Python code, use the `--pth_path` argument flag:

```bash
python Test.py --pth_path ./your_custom_path/best_model.pth --testsize 352
```

---

## 📂 Testing Output Layout & Results Verification

Upon completing the automated execution stream, the network exports the predicted segmentation matrices. The script maps the continuous feature projections into normalized outputs via a Sigmoid activation loop, casting the maps into 8-bit binary assets via `imageio`. 

The predicted outcomes are automatically sorted and serialized into the following structured topology:

```text
results/
└── MCBASegFormer/
    ├── CVC-300/                # Contains 60 predicted masks (.png)
    ├── CVC-ClinicDB/           # Contains 62 predicted masks (.png)
    ├── CVC-ColonDB/            # Contains 380 predicted masks (.png)
    ├── ETIS-LaribPolypDB/      # Contains 196 predicted masks (.png)
    └── Kvasir/                 # Contains 100 predicted masks (.png)
```

