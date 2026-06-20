# 部署与评测步骤

## 本地命令行

1. 安装环境并执行 `git lfs pull`。
2. 运行 `python scripts/verify_repository.py` 检查资产完整性。
3. 使用 `scripts/evaluate.py` 批量评测，或使用 `scripts/predict.py` 单图推理。

## 输入

- 格式：Pillow 可读取的 RGB 图像，例如 JPG、PNG、BMP、WebP。
- 二分类尺寸：缩放至 224 × 224，并归一化到均值/标准差 0.5。
- 多分类尺寸：缩放至 518 × 518，不执行额外归一化，以保持与训练流程一致。

## 输出

单图推理输出 JSON，包含预测类别、置信度和各类别概率。批量评测生成 `metrics.json`、`classification_report.txt`、`confusion_matrix.csv` 和 `predictions.csv`。

## 集成建议

生产服务中应在进程启动时加载一次模型，而不是每个请求重复加载。外部输入需限制文件大小、验证图像格式并记录模型版本与配置哈希。当前模型仅针对仓库记录的类别和数据分布验证，不应直接用于生命安全决策。
