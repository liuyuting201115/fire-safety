# fire-safety 数据集说明

## 来源与授权

- 数据集名称：`fire-safety`
- 来源一：市消防中心提供的消防安全图片。
- 来源二：通过互联网检索收集的相关图片。
- 隐私处理：图片内容已经匿名化处理。
- 使用范围：仅限获授权的比赛提交和本地研究复现。
- 再分发：不允许公开，所有图片均被 `.gitignore` 排除。
- 公开下载链接：无。

网络检索图片可能具有各自的版权限制，不因纳入本数据集而获得新的公开授权。

## 数据划分

数据使用 `scripts/prepare_dataset.py` 划分。该脚本依据原始 `01_make_three_split_dataset.py` 整理，保留以下实验规则：

- 合并原 `train` 和 `val` 图片形成各类别样本池。
- 文件路径排序后，按类别使用固定随机种子 42 打乱。
- 各类别独立按照 70% / 15% / 15% 划分 train / val / test。
- 每类样本不少于 3 张时，验证集和测试集至少各保留 1 张。

当前私有测试集统计：

| 任务 | 类别 | 数量 |
|---|---|---:|
| 二分类 | No_Hazard | 8 |
| 二分类 | Hazard | 17 |
| 多分类 | Confused_Wiring | 9 |
| 多分类 | Fire_Lane_Blocked | 10 |

## 私有目录布局

```text
data/train/binary/train|val/<类别>/*
data/train/multiclass/train|val/<类别>/*
data/test/binary/<类别>/*
data/test/multiclass/<类别>/*
data/private_manifests/*.csv
```

以上目录均不得提交到公开 Git 仓库。比赛私有提交方式见 `docs/private_submission.md`。
