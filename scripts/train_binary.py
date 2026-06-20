#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary (Hazard / No Hazard) training with independent test protocol:
- Student: DINOv2 + SE attention head (single logit)
- Teacher: DINOv2 + Linear head (single logit), loaded from output_binary/best_binary_model_sehead.pth
- Loss: (1-alpha) * BCEWithLogits + alpha * KL(student||teacher) with temperature T
- Two-stage training:
  Stage1: freeze backbone, train head only
  Stage2: unfreeze last 2 blocks + norm + head, use smaller backbone LR

IMPORTANT CHANGE:
- Force label semantics: Hazard is always positive class (1), No_Hazard is negative class (0)
  by using ImageFolder.target_transform remapping.
"""

import os
import math
import json
import random
import builtins
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

# ==============================
# CPU 兼容：强制让 xformers import 失败
# ==============================
os.environ.setdefault("XFORMERS_FORCE_DISABLE", "1")
_real_import = builtins.__import__
def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith("xformers") or name.startswith("triton"):
        raise ImportError("xformers/triton is blocked for CPU compatibility")
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _blocked_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==============================
# 配置
# ==============================
@dataclass
class TrainConfig:
    # ===== Ablation switches =====
    use_sehead: bool = True
    use_kd: bool = False
    variant_name: str = "sehead_nokd"
    out_dir_name: str = os.path.join("outputs", "binary", "sehead_nokd")

    base_dir: str = ""  # 自动设为脚本所在目录
    data_dir_name: str = os.path.join("data", "train", "binary")
    pretrain_weight_relpath: str = os.path.join("checkpoints", "dinov2_vits14_reg4_pretrain.pth")
    teacher_weight_relpath: str = os.path.join(
        "outputs",
        "binary",
        "sehead_kd_twostage",
        "best_binary_model_sehead_kd_twostage.pth",
    )

    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 0

    seed: int = 42
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    s1_epochs: int = 10
    s2_epochs: int = 20
    s1_head_lr: float = 3e-4
    s2_head_lr: float = 2e-4
    s2_backbone_lr: float = 1e-5
    early_stop_patience: int = 5

    kd_temperature: float = 2.0
    kd_alpha: float = 0.2
    kd_eps: float = 1e-6

    unfreeze_last_n_blocks: int = 2
    device: str = "cpu"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================
# 标签语义统一：Hazard 固定为正类(1)
# ==============================
def _norm_name(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")

def _is_no_hazard(name: str) -> bool:
    s = _norm_name(name)
    return ("no_hazard" in s) or ("nohazard" in s) or ("normal" in s) or ("safe" in s)

def _is_hazard(name: str) -> bool:
    if _is_no_hazard(name):
        return False
    s = _norm_name(name)
    # 只要包含 hazard 且不是 no_hazard，就视作 hazard
    return ("hazard" in s) or ("danger" in s) or ("risk" in s) or ("隐患" in s)

def load_class_names_txt(data_dir: str) -> Optional[List[str]]:
    p = os.path.join(data_dir, "class_names.txt")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        names = [x.strip() for x in f if x.strip()]
    return names or None

def build_binary_mapping(dataset: torchvision.datasets.ImageFolder, class_names: Optional[List[str]] = None) -> Dict[int, int]:
    """
    构造 old_label -> new_label 的映射，使得：
      Hazard    -> 1
      No_Hazard -> 0
    """
    classes = dataset.classes  # folder names discovered by ImageFolder
    if class_names is None:
        class_names = classes

    # 找 hazard / no_hazard 的“文件夹名”
    hazard_candidates = [n for n in class_names if n in dataset.class_to_idx and _is_hazard(n)]
    nohaz_candidates  = [n for n in class_names if n in dataset.class_to_idx and _is_no_hazard(n)]

    # 兜底：如果 class_names.txt 不规范，就从 dataset.classes 里找
    if len(hazard_candidates) != 1 or len(nohaz_candidates) != 1:
        hazard_candidates = [n for n in classes if _is_hazard(n)]
        nohaz_candidates  = [n for n in classes if _is_no_hazard(n)]

    if len(hazard_candidates) != 1 or len(nohaz_candidates) != 1:
        raise RuntimeError(
            f"[LABEL MAP ERROR] 无法唯一识别 Hazard/No_Hazard 文件夹。\n"
            f"dataset.classes={classes}\n"
            f"class_names={class_names}\n"
            f"hazard_candidates={hazard_candidates}\n"
            f"nohaz_candidates={nohaz_candidates}\n"
            f"建议把二分类文件夹命名为 Hazard 和 No_Hazard（或 class_names.txt 写清楚）。"
        )

    hazard_name = hazard_candidates[0]
    nohaz_name  = nohaz_candidates[0]

    old_hazard = dataset.class_to_idx[hazard_name]
    old_nohaz  = dataset.class_to_idx[nohaz_name]

    # new: No_Hazard=0, Hazard=1
    return {old_nohaz: 0, old_hazard: 1}

class TargetMap:
    def __init__(self, mapping: Dict[int, int]):
        self.m = dict(mapping)
    def __call__(self, y: int) -> int:
        return self.m[y]


# ==============================
# 模型：SE Head（学生）
# ==============================
class SEClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.fc2 = nn.Linear(in_channels // 16, in_channels)
        self.fc_out = nn.Linear(in_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.dim() == 4:
            avg = self.avg_pool(x).view(x.size(0), -1)
            mx = self.max_pool(x).view(x.size(0), -1)
        elif x.dim() == 3:
            L = x.size(1)
            if L > 1 and int(math.sqrt(L - 1)) ** 2 == (L - 1):
                patch_tokens = x[:, 1:, :]
            else:
                patch_tokens = x
            avg = patch_tokens.mean(dim=1)
            mx, _ = patch_tokens.max(dim=1)
        else:
            avg = mx = x

        z = self.fc2(self.relu(self.fc1(avg))) + self.fc2(self.relu(self.fc1(mx)))
        scale = torch.sigmoid(z)

        if x.dim() == 4:
            x = x * scale.unsqueeze(-1).unsqueeze(-1)
            out_feat = self.avg_pool(x).view(x.size(0), -1)
        elif x.dim() == 3:
            x = x * scale.unsqueeze(1)
            out_feat = x.mean(dim=1)
        else:
            out_feat = x * scale

        logits = self.fc_out(out_feat)  # [N,1]
        return logits


class DinoV2BinaryStudent(nn.Module):
    def __init__(self, pretrained_backbone: bool = True, use_sehead: bool = True, backbone_checkpoint: Optional[str] = None):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=pretrained_backbone and backbone_checkpoint is None,
            num_classes=0,
            img_size=518,
            dynamic_img_size=True,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        if backbone_checkpoint:
            state = torch.load(backbone_checkpoint, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        self.use_sehead = use_sehead

        if use_sehead:
            self.head = SEClassifierHead(384, 1)
        else:
            self.head = nn.Linear(384, 1)

    def _pool_features_for_linear(self, feats):
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        elif feats.dim() == 3:
            L = feats.size(1)
            if L > 1 and int(math.sqrt(L - 1)) ** 2 == (L - 1):
                feats = feats[:, 0, :]
            else:
                feats = feats.mean(dim=1)
        return feats

    def forward(self, x):
        feats = self.backbone(x)

        if self.use_sehead:
            return self.head(feats)

        feats = self._pool_features_for_linear(feats)
        return self.head(feats)

# ==============================
# 模型：线性头（教师）
# ==============================
class DinoV2BinaryTeacher(nn.Module):
    def __init__(self, pretrained_backbone: bool = True, backbone_checkpoint: Optional[str] = None):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=pretrained_backbone and backbone_checkpoint is None,
            num_classes=0,
            img_size=518,
            dynamic_img_size=True,
        )
        if backbone_checkpoint:
            state = torch.load(backbone_checkpoint, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
        self.head = nn.Linear(384, 1)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        elif feats.dim() == 3:
            L = feats.size(1)
            if L > 1 and int(math.sqrt(L - 1)) ** 2 == (L - 1):
                feats = feats[:, 0, :]
            else:
                feats = feats.mean(dim=1)
        return self.head(feats)


# ==============================
# 冻结 / 解冻
# ==============================
def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def unfreeze_last_blocks_and_norm(model: nn.Module,  n_blocks: int):
    set_backbone_trainable(model, False)

    if not hasattr(model.backbone, "blocks"):
        print("[WARN] backbone has no attribute 'blocks'. Skip partial unfreeze and keep backbone frozen.")
        return

    blocks = model.backbone.blocks
    n = len(blocks)
    n_blocks = max(0, min(n_blocks, n))
    for i in range(n - n_blocks, n):
        for p in blocks[i].parameters():
            p.requires_grad = True

    if hasattr(model.backbone, "norm"):
        for p in model.backbone.norm.parameters():
            p.requires_grad = True


# ==============================
# 数据 & 指标
# ==============================
def compute_pos_weight_from_imagefolder(ds: torchvision.datasets.ImageFolder) -> float:
    """
    pos_weight = N_neg / N_pos
    这里的 "pos" 是 label==1（我们已经强制 Hazard=1）
    """
    labels = []
    for _, old_y in ds.samples:
        y = ds.target_transform(old_y) if getattr(ds, "target_transform", None) is not None else old_y
        labels.append(int(y))

    n_pos = sum(labels)
    n_total = len(labels)
    n_neg = n_total - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    correct, total = 0, 0
    tp = fp = tn = fn = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)  # labels 已经是映射后的：Hazard=1

        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1)     # 现在 sigmoid(logit) 就是 p(hazard)
        preds = (probs >= threshold).long()

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    cm = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return acc, precision, recall, f1, cm


def build_warmup_cosine_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ==============================
# KD Loss（二分类：显式构造二类分布做 KL）
# 现在 sigmoid(logit) 语义已统一为 p(hazard)
# ==============================
def kd_kl_loss_binary(student_logits, teacher_logits, T: float, eps: float):
    teacher_prob = torch.sigmoid(teacher_logits / T)          # p(hazard)
    student_prob = torch.sigmoid(student_logits / T)          # p(hazard)

    teacher_dist = torch.cat([1 - teacher_prob, teacher_prob], dim=1)  # [p(nohaz), p(haz)]
    student_dist = torch.cat([1 - student_prob, student_prob], dim=1)

    student_log_dist = torch.log(student_dist.clamp(min=eps, max=1 - eps))
    kd = F.kl_div(student_log_dist, teacher_dist, reduction="batchmean") * (T * T)
    return kd


# ==============================
# 训练阶段
# ==============================
def train_one_stage(
    stage_name: str,
    model: nn.Module,
    teacher: Optional[nn.Module],
    use_kd: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn_bce: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    kd_alpha: float,
    kd_T: float,
    kd_eps: float,
    out_dir: str,
    patience: int,
):
    best_acc = 0.0
    bad_epochs = 0
    history: List[Dict] = []

    best_path = os.path.join(out_dir, f"best_{stage_name}.pth")
    metrics_path = os.path.join(out_dir, f"metrics_{stage_name}.json")

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"[{stage_name}] Epoch {ep}/{epochs}", ncols=110)
        for images, labels in pbar:
            images = images.to(device)
            labels_f = labels.float().unsqueeze(1).to(device)  # labels: Hazard=1

            optimizer.zero_grad(set_to_none=True)

            student_logits = model(images)
            loss_ce = loss_fn_bce(student_logits, labels_f)

            if use_kd and teacher is not None and kd_alpha > 0:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                    pt = torch.sigmoid(teacher_logits)
                    conf = (pt - 0.5).abs() * 2
                    kd_gate = (conf >= 0.6).float().mean()

                loss_kd = kd_kl_loss_binary(student_logits, teacher_logits, T=kd_T, eps=kd_eps)
                loss = (1 - kd_alpha) * loss_ce + kd_alpha * kd_gate * loss_kd
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss = loss_ce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{loss_ce.item():.4f}", kd=f"{loss_kd.item():.4f}")

        scheduler.step()

        train_loss = running / max(len(train_loader), 1)
        val_acc, val_p, val_r, val_f1, cm = evaluate(model, val_loader, device=device)

        row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "val_precision": float(val_p),
            "val_recall": float(val_r),
            "val_f1": float(val_f1),
            "cm": cm,
            "lr": [g["lr"] for g in optimizer.param_groups],
        }
        history.append(row)

        print(
            f"[{stage_name}] ep={ep}/{epochs} loss={train_loss:.4f} "
            f"| val_acc={val_acc:.4f} P={val_p:.4f} R={val_r:.4f} F1={val_f1:.4f} "
            f"| cm={cm} | lr={row['lr']}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            print(f">> Saved best: {best_path} (best_acc={best_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{stage_name}] Early stopping triggered (patience={patience}).")
                break

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_acc": best_acc, "history": history}, f, ensure_ascii=False, indent=2)

    return best_path, best_acc


def main():
    cfg = TrainConfig()
    cfg.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    data_dir = os.path.join(cfg.base_dir, cfg.data_dir_name)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    out_dir = os.path.join(cfg.base_dir, cfg.out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    transform_train = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)

    print("[INFO] Raw class_to_idx =", train_set.class_to_idx)

    # ======= 关键：标签重映射（Hazard=1, No_Hazard=0）=======
    class_names = load_class_names_txt(data_dir)  # 可有可无；有则更稳
    mapping = build_binary_mapping(train_set, class_names=class_names)
    train_set.target_transform = TargetMap(mapping)
    val_set.target_transform = TargetMap(mapping)

    # 打印映射结果
    inv = {v: k for k, v in train_set.class_to_idx.items()}
    print("[INFO] Label mapping (old->new) =", mapping)
    print(f"[INFO] Meaning after mapping: new=1 is Hazard, new=0 is No_Hazard")
    print(f"[INFO] Example folders: old label {list(mapping.keys())[0]}='{inv[list(mapping.keys())[0]]}', old label {list(mapping.keys())[1]}='{inv[list(mapping.keys())[1]]}'")

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    pos_weight = compute_pos_weight_from_imagefolder(train_set)
    print(f"[INFO] train={len(train_set)}, val={len(val_set)}, pos_weight(Hazard)= {pos_weight:.4f}")

    pretrain_path = os.path.join(cfg.base_dir, cfg.pretrain_weight_relpath)
    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError(f"Missing pretrained backbone: {pretrain_path}")

    student = DinoV2BinaryStudent(
        pretrained_backbone=True,
        use_sehead=cfg.use_sehead,
        backbone_checkpoint=pretrain_path,
    ).to(device)

    teacher = None

    if cfg.use_kd:
        teacher = DinoV2BinaryTeacher(
            pretrained_backbone=True,
            backbone_checkpoint=pretrain_path,
        ).to(device)
        teacher_path = os.path.join(cfg.base_dir, cfg.teacher_weight_relpath)

        if os.path.exists(teacher_path):
            state = torch.load(teacher_path, map_location=device)
            teacher.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded teacher weights: {teacher_path}")
        else:
            print(f"[WARN] Teacher weights not found: {teacher_path}")
            print("[WARN] KD will run with an untrained linear teacher, which is not recommended.")

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # 现在 pos_weight 是针对 Hazard(1) 的，语义正确
    loss_fn_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # ===== Stage 1 =====
    print("\n======================")
    print("Stage1: freeze backbone, train head only")
    print("======================")

    set_backbone_trainable(student, False)
    for p in student.head.parameters():
        p.requires_grad = True

    warmup1 = min(3, (cfg.s1_epochs // 2) or 1)
    optim_s1 = torch.optim.AdamW(
        [{"params": student.head.parameters(), "lr": cfg.s1_head_lr}],
        weight_decay=cfg.weight_decay
    )
    sched_s1 = build_warmup_cosine_scheduler(optim_s1, total_epochs=cfg.s1_epochs, warmup_epochs=warmup1)

    best_s1_path, best_s1_acc = train_one_stage(
        stage_name="stage1",
        model=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_fn_bce=loss_fn_bce,
        optimizer=optim_s1,
        scheduler=sched_s1,
        epochs=cfg.s1_epochs,
        kd_alpha=cfg.kd_alpha,
        kd_T=cfg.kd_temperature,
        kd_eps=cfg.kd_eps,
        out_dir=out_dir,
        patience=cfg.early_stop_patience,
        use_kd=cfg.use_kd,
    )

    if os.path.exists(best_s1_path):
        student.load_state_dict(torch.load(best_s1_path, map_location=device), strict=False)

    # ===== Stage 2 =====
    print("\n======================")
    print("Stage2: unfreeze last blocks + norm + head")
    print("======================")

    unfreeze_last_blocks_and_norm(student, cfg.unfreeze_last_n_blocks)
    for p in student.head.parameters():
        p.requires_grad = True

    head_params = [p for p in student.head.parameters() if p.requires_grad]
    backbone_params = [p for p in student.backbone.parameters() if p.requires_grad]

    warmup2 = min(5, (cfg.s2_epochs // 3) or 1)
    optim_s2 = torch.optim.AdamW(
        [
            {"params": head_params, "lr": cfg.s2_head_lr},
            {"params": backbone_params, "lr": cfg.s2_backbone_lr},
        ],
        weight_decay=cfg.weight_decay
    )
    sched_s2 = build_warmup_cosine_scheduler(optim_s2, total_epochs=cfg.s2_epochs, warmup_epochs=warmup2)

    best_s2_path, best_s2_acc = train_one_stage(
        stage_name="stage2",
        model=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_fn_bce=loss_fn_bce,
        optimizer=optim_s2,
        scheduler=sched_s2,
        epochs=cfg.s2_epochs,
        kd_alpha=cfg.kd_alpha,
        kd_T=cfg.kd_temperature,
        kd_eps=cfg.kd_eps,
        out_dir=out_dir,
        patience=cfg.early_stop_patience,
        use_kd=cfg.use_kd,
    )

    final_path = os.path.join(out_dir, "best_binary_model_sehead_kd_twostage.pth")
    if os.path.exists(best_s2_path):
        torch.save(torch.load(best_s2_path, map_location="cpu"), final_path)
        print(f"\n[DONE] Final best model saved to: {final_path}")
        print(f"[DONE] best_stage1_acc={best_s1_acc:.4f}, best_stage2_acc={best_s2_acc:.4f}")
    else:
        print("\n[WARN] Stage2 best checkpoint not found; stage1 best is kept.")
        print(f"[WARN] Stage1 best: {best_s1_path}")


if __name__ == "__main__":
    main()
