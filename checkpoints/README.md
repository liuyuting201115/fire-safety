# 模型权重

| 文件 | 用途 |
|---|---|
| `binary_best.pth` | 二分类最终模型 |
| `multiclass_best.pth` | 多分类最终模型 |
| `dinov2_vits14_reg4_pretrain.pth` | 重新训练使用的 DINOv2 ViT-S/14 预训练骨干 |

这些文件通过 Git LFS 管理，完整 SHA-256 记录在 `manifests/checkpoints.sha256`。发布前应再次核对 DINOv2 官方权重许可，并在 Release 页面同时提供文件大小和校验值。
