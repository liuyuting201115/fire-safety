# 比赛私有提交说明

公共 GitHub 仓库只包含代码、配置、模型权重、汇总指标和文档，不包含任何 `fire-safety` 图片。

若赛事主办方要求提交完整可验证材料，应通过赛事指定的私有上传渠道提供以下内容：

```text
data/test/binary/<类别>/*
data/test/multiclass/<类别>/*
data/private_manifests/dataset_split_summary.csv
data/private_manifests/split_manifest.csv
results/reference/*
```

提交前执行：

```bash
python scripts/verify_repository.py --require-private-data
python scripts/evaluate.py --config configs/binary.yaml
python scripts/evaluate.py --config configs/multiclass.yaml
```

私有压缩包及传输链接应设置访问权限和有效期。不得将数据图片、原始文件名清单、逐样本预测结果或带图片路径的日志上传到公开 GitHub、公开网盘或公开 Release。
