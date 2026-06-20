# 运行环境

## 已验证平台

- 验证日期：2026-06-20
- 操作系统：Microsoft Windows NT 10.0.26200.0，64 位
- Python：3.12.11
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB
- NVIDIA 驱动：591.86
- CUDA Runtime：12.8

## 核心依赖

| 软件 | 版本 |
|---|---|
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| timm | 1.0.19 |
| numpy | 2.1.2 |
| pandas | 3.0.2 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.6 |
| PyYAML | 6.0.2 |
| Pillow | 12.2.0 |

`requirements.txt` 固定 Python 包版本。CUDA 驱动和 PyTorch wheel 应按目标平台单独选择。评测支持 CPU，但速度显著慢于 GPU。

## 可复现设置

- 随机种子：42
- 评测阶段关闭梯度并启用 `model.eval()`
- DataLoader 不打乱测试集，默认 `num_workers=0`
- 评测配置和类别顺序保存在 `configs/`
- 权重和测试图片使用 SHA-256 校验
