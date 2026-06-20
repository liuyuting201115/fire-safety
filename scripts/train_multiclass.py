# -*- coding: utf-8 -*-
"""
多分类（高准确率优先版）+ SEHead + Stage2 KD（跨阶段自蒸馏）
- 训练集使用随机增强，验证集使用确定性预处理，避免验证结果被随机增强影响
- 无类别权重、无Label Smoothing、无Dropout/DropPath、无EMA
- 两阶段微调：S1只训head；S2解冻最后2个block + norm + head
- S2阶段引入知识蒸馏：用S1最佳模型作为Teacher，指导S2微调（稳定表示漂移）
- 以 val_acc 作为早停与最优保存指标
"""
import os, random, math, copy
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset
import timm
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========= 基本配置 =========
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, "data", "train", "multiclass")
USE_SEHEAD = True
KD_ENABLE_STAGE2 = False

VARIANT_NAME = "sehead_kd" if (USE_SEHEAD and KD_ENABLE_STAGE2) else \
               "sehead_nokd" if USE_SEHEAD else \
               "linear_nokd"

OUT_DIR = os.path.join(BASE, "outputs", "multiclass", VARIANT_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

SEED       = 42
device     = torch.device("cpu")  # 保留你原写法；需要GPU可自行改成cuda自动
s1_epochs  = 10
s2_epochs  = 20
batch_size = 24
img_size   = 518

# 学习率 & 早停
s1_head_lr      = 3e-4
s2_head_lr      = 2e-4
s2_backbone_lr  = 1e-5
weight_decay    = 0.05
patience        = 5
label_smooth    = 0.0

MAX_PER_CLASS_TRAIN = None
MAX_PER_CLASS_VAL   = None

pretrain_ckpt = os.path.join(BASE, "checkpoints", "dinov2_vits14_reg4_pretrain.pth")

# ===== KD 超参（只在stage2使用）=====
KD_ALPHA = 0.5
KD_T     = 4.0
KD_WARMUP_RATIO = 0.3   # 前30% epoch 从0爬升到 kd_alpha，后面固定
TEACHER_ON_CPU = True

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ========= 数据增强：train 随机增强，val 确定性预处理 =========
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.0),
                                 ratio=(0.9, 1.1), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
])

# ========= 类名 =========
cls_txt = os.path.join(DATA_DIR, "class_names.txt")
assert os.path.exists(cls_txt), f"缺少 {cls_txt}"
with open(cls_txt, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]
num_classes = len(class_names); assert num_classes >= 2

# ========= 数据集 & 标签对齐 =========
transform_val = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

train_full = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform_train)
val_full   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=transform_val)

def build_mapping(dataset, class_names):
    missing = [n for n in class_names if n not in dataset.class_to_idx]
    extra   = [n for n in dataset.classes  if n not in class_names]
    if missing:
        raise RuntimeError(f"在数据集中找不到以下类别文件夹：{missing}")
    old_to_new = { dataset.class_to_idx[n]: i for i, n in enumerate(class_names) }
    class TargetMap:
        def __init__(self, m): self.m = dict(m)
        def __call__(self, y):  return self.m[y]
    dataset.target_transform = TargetMap(old_to_new)
    if extra: print(f"[WARN] 发现额外类别（将被忽略）：{extra}")

build_mapping(train_full, class_names)
build_mapping(val_full,   class_names)

def build_limited_subset(dataset, max_per_class=None):
    idx_all = list(range(len(dataset.samples))); random.shuffle(idx_all)
    kept, per_cnt = [], {i:0 for i in range(len(class_names))}
    for idx in idx_all:
        old_y = dataset.samples[idx][1]
        new_y = dataset.target_transform(old_y)
        if (max_per_class is None) or (per_cnt[new_y] < max_per_class):
            kept.append(idx); per_cnt[new_y]+=1
    return Subset(dataset, kept)

train_dataset = build_limited_subset(train_full, MAX_PER_CLASS_TRAIN)
val_dataset   = build_limited_subset(val_full,   MAX_PER_CLASS_VAL)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

print("[INFO] 类别顺序：", class_names)
print("[INFO] 训练样本数：", len(train_dataset), "验证样本数：", len(val_dataset))


# ==============================
#  SEHead（GAP+GMP融合 + SE通道注意力）
# ==============================
class SEHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, reduction: int = 16):
        super().__init__()
        hid = max(1, (2 * in_dim) // reduction)
        self.fc1 = nn.Linear(2 * in_dim, hid, bias=True)
        self.fc2 = nn.Linear(hid, 2 * in_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.cls = nn.Linear(2 * in_dim, num_classes, bias=True)

    def forward(self, tokens: torch.Tensor, num_prefix_tokens: int = 1):
        if tokens.dim() == 2:
            avg = tokens
            mx  = tokens
        else:
            if tokens.size(1) > num_prefix_tokens:
                patch = tokens[:, num_prefix_tokens:, :]
            else:
                patch = tokens
            avg = patch.mean(dim=1)
            mx  = patch.amax(dim=1)
        x = torch.cat([avg, mx], dim=1)

        w = self.act(self.fc1(x))
        w = torch.sigmoid(self.fc2(w))
        x = x * w
        return self.cls(x)


class LinearTokenHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.cls = nn.Linear(in_dim, num_classes)

    def forward(self, tokens: torch.Tensor, num_prefix_tokens: int = 1):
        if tokens.dim() == 2:
            feat = tokens
        else:
            if tokens.size(1) > num_prefix_tokens:
                # 使用 CLS token 作为 backbone-only linear probe 的表征
                feat = tokens[:, 0, :]
            else:
                feat = tokens.mean(dim=1)
        return self.cls(feat)


class DinoV2WithHead(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, use_sehead: bool = True):
        super().__init__()
        self.backbone = backbone
        self.blocks = backbone.blocks
        self.norm = backbone.norm
        self.use_sehead = use_sehead

        if hasattr(backbone, "embed_dim"):
            embed_dim = backbone.embed_dim
        elif hasattr(backbone, "num_features"):
            embed_dim = backbone.num_features
        else:
            embed_dim = 384

        if use_sehead:
            self.head = SEHead(embed_dim, num_classes=num_classes, reduction=16)
        else:
            self.head = LinearTokenHead(embed_dim, num_classes=num_classes)

    def forward_features(self, x):
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            for k in ["x", "last_hidden_state", "features"]:
                if k in out:
                    return out[k]
            return list(out.values())[0]
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out

    def forward(self, x):
        tokens = self.forward_features(x)
        num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        return self.head(tokens, num_prefix_tokens=num_prefix)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int):
        if hasattr(self.backbone, "embed_dim"):
            embed_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, "num_features"):
            embed_dim = self.backbone.num_features
        else:
            embed_dim = 384

        if self.use_sehead:
            self.head = SEHead(embed_dim, num_classes=num_classes, reduction=16)
        else:
            self.head = LinearTokenHead(embed_dim, num_classes=num_classes)


# ========= 构建骨干 + 加载预训练 =========
backbone = timm.create_model(
    "vit_small_patch14_dinov2",
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.0
).to(device)

if os.path.exists(pretrain_ckpt):
    print(f"[INFO] 读取预训练：{pretrain_ckpt}")
    state_dict = torch.load(pretrain_ckpt, map_location="cpu")
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    print(f"[CHECK] pretrain missing={len(missing)} unexpected={len(unexpected)}")
else:
    print("[WARN] 未找到预训练权重，将从随机初始化开始")

model = DinoV2WithHead(
    backbone,
    num_classes=num_classes,
    use_sehead=USE_SEHEAD
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

def evaluate():
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred) * 100.0
    rpt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return acc, rpt, cm

def build_scheduler(optimizer, total_epochs, warmup_epochs=3):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def kd_loss_multi(logits_s, logits_t, T=4.0):
    ps = torch.log_softmax(logits_s / T, dim=1)
    pt = torch.softmax(logits_t / T, dim=1)
    return nn.functional.kl_div(ps, pt, reduction="batchmean") * (T * T)

def train_stage(stage_name, epochs, optimizer, scheduler,
                best_path, report_path, cm_path, curve_path,
                teacher_model=None, kd_enable=False, kd_alpha=0.5, kd_T=4.0, teacher_on_cpu=False):
    history, best_acc, bad_epochs = [], 0.0, 0

    if teacher_model is not None and kd_enable:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        t_device = torch.device("cpu") if teacher_on_cpu else device
        teacher_model.to(t_device)
    else:
        t_device = None

    for ep in range(1, epochs + 1):

        # ===== 渐进式 KD 权重：前 warmup_epochs 线性爬升，之后固定 =====
        alpha_now = 0.0
        if (teacher_model is not None) and kd_enable:
            warmup_epochs = max(1, int(epochs * KD_WARMUP_RATIO))
            alpha_now = kd_alpha * min(1.0, ep / warmup_epochs)

        model.train()
        run_loss = 0.0

        with tqdm(total=len(train_loader),
                  desc=f"[{stage_name}] Epoch {ep}/{epochs}",
                  unit="batch") as pbar:
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                logits_s = model(imgs)
                loss_ce = criterion(logits_s, labels)

                if (teacher_model is not None) and kd_enable:
                    with torch.no_grad():
                        if t_device != device:
                            t_logits = teacher_model(imgs.detach().to(t_device)).to(device)
                        else:
                            t_logits = teacher_model(imgs)

                    # ✅ 多分类蒸馏：KL(softmax(s/T), softmax(t/T)) * T^2
                    loss_kd = kd_loss_multi(logits_s, t_logits, T=kd_T)

                    # ✅ 使用渐进式 alpha_now
                    loss = (1 - alpha_now) * loss_ce + alpha_now * loss_kd
                else:
                    loss = loss_ce

                loss.backward()
                optimizer.step()

                run_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}", alpha=f"{alpha_now:.2f}")
                pbar.update(1)

        train_loss = run_loss / max(1, len(train_loader))
        val_acc, rpt, cm = evaluate()
        scheduler.step()
        print(f"[{stage_name}] alpha_now={alpha_now:.3f} | train_loss={train_loss:.4f} | val_acc={val_acc:.2f}%")
        history.append({"epoch": ep, "train_loss": train_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            torch.save(model.state_dict(), best_path)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(rpt)
            pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path, encoding="utf-8-sig")
            print(f"[{stage_name}] [BEST] {best_acc:.2f}% -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{stage_name}] [EARLY STOP] 连续 {patience} 轮未提升，提前停止。")
                break

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(OUT_DIR, f"{stage_name}_history.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.plot(df["epoch"], df["train_loss"], marker="o"); plt.title(f"{stage_name} Loss"); plt.grid()
    plt.subplot(1, 2, 2); plt.plot(df["epoch"], df["val_acc"],   marker="o"); plt.title(f"{stage_name} Val Acc");  plt.grid()
    plt.tight_layout(); plt.savefig(curve_path, dpi=200); plt.close()
    return best_acc


# ===== 阶段一：只训练分类头 =====
for p in model.parameters(): p.requires_grad = False
for p in model.get_classifier().parameters(): p.requires_grad = True

optimizer_s1 = optim.AdamW(
    [{"params": [p for p in model.get_classifier().parameters() if p.requires_grad], "lr": s1_head_lr}],
    weight_decay=weight_decay
)
scheduler_s1 = build_scheduler(optimizer_s1, total_epochs=s1_epochs, warmup_epochs=min(3, s1_epochs // 2 or 1))

s1_best_path = os.path.join(OUT_DIR, "best_stage1.pth")
train_stage(
    "stage1_head_only", s1_epochs, optimizer_s1, scheduler_s1,
    best_path=s1_best_path,
    report_path=os.path.join(OUT_DIR, "val_report_stage1.txt"),
    cm_path=os.path.join(OUT_DIR, "val_confusion_matrix_stage1.csv"),
    curve_path=os.path.join(OUT_DIR, "train_curves_stage1.png"),
    teacher_model=None, kd_enable=False
)

# 用阶段一最佳作为阶段二起点
if os.path.exists(s1_best_path):
    model.load_state_dict(torch.load(s1_best_path, map_location="cpu"))

# ===== 构造Teacher：使用S1最佳模型作为教师（跨阶段自蒸馏）=====
teacher_model = copy.deepcopy(model) if KD_ENABLE_STAGE2 else None

# ===== 阶段二：解冻最后2个 block + norm + head =====
for p in model.parameters(): p.requires_grad = False
for blk in model.blocks[-2:]:
    for p in blk.parameters(): p.requires_grad = True
for p in model.norm.parameters(): p.requires_grad = True
for p in model.get_classifier().parameters(): p.requires_grad = True

param_groups = [
    {"params": [p for p in model.get_classifier().parameters() if p.requires_grad], "lr": s2_head_lr},
    {"params": [p for blk in model.blocks[-2:] for p in blk.parameters() if p.requires_grad], "lr": s2_backbone_lr},
    {"params": [p for p in model.norm.parameters() if p.requires_grad], "lr": s2_backbone_lr},
]
optimizer_s2 = optim.AdamW(param_groups, weight_decay=weight_decay)
scheduler_s2 = build_scheduler(optimizer_s2, total_epochs=s2_epochs, warmup_epochs=min(5, s2_epochs // 3 or 1))

best_path_final = os.path.join(OUT_DIR, "best_multi_model.pth")
train_stage(
    "stage2_last2_and_head", s2_epochs, optimizer_s2, scheduler_s2,
    best_path=best_path_final,
    report_path=os.path.join(OUT_DIR, "val_report_stage2.txt"),
    cm_path=os.path.join(OUT_DIR, "val_confusion_matrix_stage2.csv"),
    curve_path=os.path.join(OUT_DIR, "train_curves_stage2.png"),
    teacher_model=teacher_model,
    kd_enable=KD_ENABLE_STAGE2,
    kd_alpha=KD_ALPHA,
    kd_T=KD_T,
    teacher_on_cpu=TEACHER_ON_CPU
)

print(f"[DONE] 两阶段完成（SEHead + Stage2 KD），最终权重：{best_path_final}")
