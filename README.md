# Fire Safety — DINOv2 Classification

基于 DINOv2 ViT-S/14 的消防隐患图像分类项目，包含二分类、多分类、最终模型、确定性数据划分脚本、独立评测流程和参考结果。

公开仓库地址：[liuyuting201115/fire-safety](https://github.com/liuyuting201115/fire-safety)。由于数据提供协议及网络图片再分发限制，本仓库不公开任何训练、验证或测试图片；如要求提交完整可验证材料，可以提供比赛私有提交包附带数据以完成复现。

## 已验证结果

| 任务 | 私有测试样本 | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Hazard / No_Hazard | 25 | 0.960000 | 0.955437 |
| Confused_Wiring / Fire_Lane_Blocked | 19 | 1.000000 | 1.000000 |

以上结果于 2026-06-20 使用私有测试集、仓库内最终权重及 `scripts/evaluate.py` 重新计算，并通过配置中的严格数值校验。详细结果见 `results/reference/metrics.json`。

## 仓库结构

```text
.
├── checkpoints/               # Git LFS：最终模型及预训练权重
├── configs/                   # 二分类、多分类复现配置
├── data/
│   ├── README.md              # 数据来源、授权和目录协议
│   ├── train/                 # 私有，不进入 Git
│   └── test/                  # 私有，不进入 Git
├── docs/                      # 环境、部署、私有交付和模型说明
├── manifests/checkpoints.sha256
├── results/reference/         # 可公开的汇总指标与混淆矩阵
├── scripts/                   # 数据划分、训练、评测、推理、校验
├── src/fire_hazard/           # 可复用模型、数据和评测代码
├── tests/                     # 公共仓库快速测试
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## 1. 获取代码和模型

```bash
git clone https://github.com/liuyuting201115/fire-safety.git
cd fire-safety
git lfs pull
```

模型由 Git LFS 管理。若比赛平台通过压缩包提交，请确认 `checkpoints/*.pth` 是实际模型文件，而不是 LFS 指针。

## 2. 安装环境

推荐 Python 3.12。已验证 GPU 环境使用 PyTorch 2.8.0、CUDA 12.8：

```bash
conda create -n fire-safety python=3.12 -y
conda activate fire-safety
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

CPU 环境将索引改为 `https://download.pytorch.org/whl/cpu`。完整平台信息见 `docs/environment.md`。

## 3. 校验公共交付文件

```bash
python scripts/verify_repository.py
python -m unittest discover -s tests -v
```

该步骤校验三个 LFS 权重及公共代码，不要求私有图片存在。

## 4. 准备私有数据

原始数据目录应包含：

```text
<SOURCE_ROOT>/fire_image/train|val/<类别>/*
<SOURCE_ROOT>/fire_image_binary/train|val/<类别>/*
```

使用与比赛实验一致的 seed 42、70%/15%/15% 分层划分：

```bash
python scripts/prepare_dataset.py --source-root <SOURCE_ROOT> --output-root data --overwrite
python scripts/verify_repository.py --require-private-data
```

划分清单写入被 Git 忽略的 `data/private_manifests/`，其中含原文件名，不得公开。

## 5. 复现私有测试结果

```bash
python scripts/evaluate.py --config configs/binary.yaml
python scripts/evaluate.py --config configs/multiclass.yaml
```

成功时分别输出：

```text
Accuracy: 0.960000    Macro F1: 0.955437    Reference verification: PASS
Accuracy: 1.000000    Macro F1: 1.000000    Reference verification: PASS
```

逐样本预测、分类报告、混淆矩阵和机器可读指标写入被 Git 忽略的 `results/reproduced/`。指标偏离参考值时脚本以非零状态退出。

## 6. 推理与训练

```bash
python scripts/predict.py --config configs/binary.yaml --image path/to/image.jpg
python scripts/train_binary.py
python scripts/train_multiclass.py
```

训练脚本采用随机种子 42 和两阶段训练策略。训练耗时和最终数值可能因硬件、CUDA 算子及数据版本产生小幅差异。

## 指标定义

- Accuracy：预测正确的样本比例。
- Macro F1：各类别 F1 的无权平均。
- 二分类正类固定为 `Hazard`，决策阈值为 0.5。

## 数据与许可证

数据集名称为 `fire-safety`，由市消防中心提供的匿名化图片及网络检索图片组成，仅限获授权的比赛和本地研究使用，不允许通过本公开仓库再分发。代码采用 MIT License；DINOv2 等第三方组件遵守其上游许可证，详见 `THIRD_PARTY_NOTICES.md`。
