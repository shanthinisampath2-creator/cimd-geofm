# Plain-Python export of notebooks/01_WHU-OPT-SAR_dual_stream_CNN.ipynb, for reading and review.
# Cells are separated by "# %%" markers (VS Code / Spyder / Jupytext cell format).
# The notebook is the reference version; run cells in order. Training cells are resume-safe.


# %% [markdown]
# # CIMD-GeoFM v2 — GroupNorm retraining of A0–A3 on WHU-OPT-SAR
#
# **What changed from v1**
# 1. **GroupNorm instead of BatchNorm**, so the SAR-zeroed and optical-zeroed training passes can no longer corrupt the normalisation statistics (confirmed by the BN-recalibration diagnostic).
# 2. **One-time patch cache**: each large TIFF is read once, normalised per scene, and cut into 256×256 patches. Training never opens a TIFF again.
# 3. **Class-weighted cross-entropy**, so rare classes (road, others) receive gradient.
# 4. **Resume-safe training**: a checkpoint is saved every epoch, finished runs are skipped, and early stopping is applied.
# 5. **Evaluation never depends on variables in memory**, so it works after any kernel restart.
#
# **Run order**
# - Every session: run Cells 1–5 (setup). The first run of Cell 3 builds the cache, which takes about 5–10 minutes. Later runs load it in seconds.
# - Training: run Cells 6–9 one at a time. If one is interrupted, rerun it and it resumes.
# - Results: Cells 10–13 can be run at any time after Cells 1–5.

# %%
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/conda OpenMP clash guard (see README)
# ============================================================
# CELL 1 — IMPORTS, PATHS, CONFIGURATION
# ============================================================
import os, sys, json, time, random, string, shutil, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display

import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Python :", sys.executable)
print("PyTorch:", torch.__version__, "| Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

# ---------- locate the extracted WHU-OPT-SAR data ----------
# Leave as None to search C:\content, F:\content, ... automatically.
DATA_DIR_OVERRIDE = None          # e.g. r"C:\content\WHU_OPT_SAR_EXTRACTED"

def find_folder(rel, must_contain):
    cands = [Path("/" + rel)] + [Path(f"{d}:/{rel}") for d in string.ascii_uppercase]
    for c in cands:
        try:
            if c.exists() and any(c.rglob(must_contain)):
                return c
        except OSError:
            pass
    return None

EXTRACT_DIR = Path(DATA_DIR_OVERRIDE) if DATA_DIR_OVERRIDE else find_folder("content/WHU_OPT_SAR_EXTRACTED", "*.tif")
if EXTRACT_DIR is None or not EXTRACT_DIR.exists():
    raise RuntimeError("WHU_OPT_SAR_EXTRACTED not found. Set DATA_DIR_OVERRIDE to the folder path.")

ROOT = EXTRACT_DIR.parent
OLD_RESULTS = ROOT / "CIMD_GeoFM_WHU_RESULTS"          # v1 results (only its scene split is reused)
OUT_DIR = ROOT / "CIMD_GeoFM_WHU_RESULTS_GN"            # v2 results (new folder, v1 is left untouched)
CKPT_DIR = OUT_DIR / "checkpoints"
CACHE_DIR = ROOT / "WHU_PATCH_CACHE_256"
for p in [OUT_DIR, CKPT_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Data   :", EXTRACT_DIR.resolve())
print("Results:", OUT_DIR.resolve())
print("Cache  :", CACHE_DIR.resolve())

# ---------- data ----------
PATCH_SIZE = 256
TRAIN_CROPS_PER_SCENE = 24        # random crops cached per training scene
VAL_STRIDE = 768                  # grid tiles for validation
TEST_STRIDE = 512                 # grid tiles for testing (systematic coverage of every test scene)
MIN_VALID_FRACTION = 0.05         # skip patches that are almost entirely unlabelled

NUM_CLASSES = 7
IGNORE_INDEX = 255
CLASS_NAMES = ["farmland", "city", "village", "water", "forest", "road", "others"]
RAW_TO_CLASS = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6}

# ---------- training ----------
SEEDS = [42, 43, 44]              # paper setting: 3 seeds, report mean ± std
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 25
PATIENCE = 5                      # early stopping on the validation robust score
LR = 3e-4
WEIGHT_DECAY = 1e-4
USE_AMP = (DEVICE == "cuda")      # mixed precision: about 1.5–2x faster on GPU
USE_CLASS_WEIGHTS = True
GN_GROUPS = 8

# ---------- CIMD coefficients (unchanged from v1) ----------
ALPHA = 0.50   # missing-modality segmentation
BETA  = 0.20   # dependency estimation
GAMMA = 0.30   # DWCR
DELTA = 0.10   # dependency-guided compensation
TAU   = 0.05   # bounded dependency temperature

EXPERIMENTS = {
    "A0": "Full-modality baseline",
    "A1": "Missing-modality supervised training",
    "A2": "Uniform consistency",
    "A3": "CIMD + DWCR (proposed)",
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(42)

# %%
# ============================================================
# CELL 2 — MATCH SCENES + SCENE-LEVEL SPLIT (same split as v1)
# ============================================================
groups = defaultdict(list)
for p in sorted(EXTRACT_DIR.rglob("*.tif")):
    groups[p.parent.name.lower()].append(p)

OPT = {p.stem: p for p in groups["optical"]}
SAR = {p.stem: p for p in groups["sar"]}
LBL = {p.stem: p for p in groups["lbl"]}
common_ids = sorted(set(OPT) & set(SAR) & set(LBL))
print("Optical / SAR / Label files:", len(OPT), len(SAR), len(LBL))
print("Matched scenes:", len(common_ids))

split_csv_old = OLD_RESULTS / "scene_split.csv"
split_csv_new = OUT_DIR / "scene_split.csv"

if split_csv_new.exists():
    split_df = pd.read_csv(split_csv_new)
    print("Using the split already saved for v2.")
elif split_csv_old.exists():
    split_df = pd.read_csv(split_csv_old)
    print("Reusing the v1 split, so v1 and v2 test on the same scenes.")
else:
    valid_ids = []
    for sid in common_ids:
        with rasterio.open(OPT[sid]) as ro, rasterio.open(SAR[sid]) as rs, rasterio.open(LBL[sid]) as rl:
            if (ro.height, ro.width) == (rs.height, rs.width) == (rl.height, rl.width):
                valid_ids.append(sid)
    ids = np.array(valid_ids)
    np.random.default_rng(42).shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(0.20 * n)))
    n_val = max(1, int(round(0.16 * n)))
    n_train = n - n_val - n_test
    split_df = pd.DataFrame(
        [{"scene": s, "split": "train"} for s in ids[:n_train]] +
        [{"scene": s, "split": "val"} for s in ids[n_train:n_train + n_val]] +
        [{"scene": s, "split": "test"} for s in ids[n_train + n_val:]])
    print("Created a new 64/16/20 split.")

split_df["scene"] = split_df["scene"].astype(str)
split_df.to_csv(split_csv_new, index=False)
SPLIT = {k: split_df.loc[split_df["split"] == k, "scene"].tolist() for k in ["train", "val", "test"]}
missing = [s for s in split_df["scene"] if s not in common_ids]
if missing:
    raise RuntimeError(f"{len(missing)} split scenes have no matching TIFFs, e.g. {missing[:3]}")
print({k: len(v) for k, v in SPLIT.items()})

# %%
# ============================================================
# CELL 3 — ONE-TIME PATCH CACHE (loads instantly on later runs)
# ============================================================
LABEL_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for raw, cls in RAW_TO_CLASS.items():
    LABEL_LUT[raw] = cls

def convert_label(y):
    y = y.astype(np.int64)
    out = np.full(y.shape, IGNORE_INDEX, dtype=np.uint8)
    ok = (y >= 0) & (y < 256)
    out[ok] = LABEL_LUT[y[ok]]
    return out

def scene_to_uint8(arr):
    """Per-scene 2–98 percentile normalisation, stored as uint8 to save memory."""
    arr = arr.astype(np.float32)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for c in range(arr.shape[0]):
        sub = arr[c, ::4, ::4]
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            continue
        lo, hi = np.percentile(sub, [2, 98])
        if hi > lo:
            x = np.clip((np.nan_to_num(arr[c], nan=lo) - lo) / (hi - lo), 0, 1)
            out[c] = np.round(x * 255).astype(np.uint8)
    return out

def load_scene(sid):
    with rasterio.open(OPT[sid]) as s: o = s.read()
    with rasterio.open(SAR[sid]) as s: r = s.read()
    with rasterio.open(LBL[sid]) as s: y = s.read(1)
    if not (o.shape[1:] == r.shape[1:] == y.shape):
        raise RuntimeError(f"Scene {sid} is not aligned: {o.shape}, {r.shape}, {y.shape}")
    return scene_to_uint8(o), scene_to_uint8(r), convert_label(y)

def grid_positions(H, W, stride):
    rows = list(range(0, H - PATCH_SIZE + 1, stride))
    cols = list(range(0, W - PATCH_SIZE + 1, stride))
    return [(r, c) for r in rows for c in cols]

def random_positions(y, n, rng, tries_per_patch=30):
    H, W = y.shape
    pos = []
    for _ in range(n):
        best = None
        for _ in range(tries_per_patch):
            r = int(rng.integers(0, H - PATCH_SIZE + 1))
            c = int(rng.integers(0, W - PATCH_SIZE + 1))
            best = (r, c)
            if np.mean(y[r:r + PATCH_SIZE, c:c + PATCH_SIZE] != IGNORE_INDEX) >= MIN_VALID_FRACTION:
                break
        pos.append(best)
    return pos

CACHE_CONFIG = {"patch": PATCH_SIZE, "train_crops": TRAIN_CROPS_PER_SCENE, "val_stride": VAL_STRIDE,
                "test_stride": TEST_STRIDE, "min_valid": MIN_VALID_FRACTION, "split": SPLIT}

def build_split_cache(split):
    path = CACHE_DIR / f"{split}.npz"
    meta_path = CACHE_DIR / f"{split}_meta.json"
    cfg = {k: v for k, v in CACHE_CONFIG.items() if k != "split"}
    cfg["scenes"] = SPLIT[split]
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == cfg:
        d = np.load(path)
        print(f"{split:5s}: loaded {len(d['lbl'])} cached patches")
        return {k: d[k] for k in d.files}

    rng = np.random.default_rng(1234 + ["train", "val", "test"].index(split))
    O, S, Y, SC, RC = [], [], [], [], []
    for si, sid in enumerate(tqdm(SPLIT[split], desc=f"Caching {split}")):
        o, s, y = load_scene(sid)
        H, W = y.shape
        if H < PATCH_SIZE or W < PATCH_SIZE:
            raise RuntimeError(f"Scene {sid} is smaller than {PATCH_SIZE}px")
        if split == "train":
            pos = random_positions(y, TRAIN_CROPS_PER_SCENE, rng)
        else:
            pos = grid_positions(H, W, VAL_STRIDE if split == "val" else TEST_STRIDE)
        for r, c in pos:
            yy = y[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
            if split != "train" and np.mean(yy != IGNORE_INDEX) < MIN_VALID_FRACTION:
                continue
            O.append(o[:, r:r + PATCH_SIZE, c:c + PATCH_SIZE])
            S.append(s[:, r:r + PATCH_SIZE, c:c + PATCH_SIZE])
            Y.append(yy)
            SC.append(si); RC.append((r, c))
        del o, s, y
    d = {"opt": np.stack(O), "sar": np.stack(S), "lbl": np.stack(Y),
         "scene_idx": np.array(SC, dtype=np.int32), "rowcol": np.array(RC, dtype=np.int32)}
    np.savez(path, **d)
    meta_path.write_text(json.dumps(cfg))
    print(f"{split:5s}: built {len(d['lbl'])} patches -> {path.name}")
    return d

t0 = time.time()
CACHE = {sp: build_split_cache(sp) for sp in ["train", "val", "test"]}
OPT_CH = CACHE["train"]["opt"].shape[1]
SAR_CH = CACHE["train"]["sar"].shape[1]
print(f"Optical channels: {OPT_CH} | SAR channels: {SAR_CH} | {time.time() - t0:.0f}s")

# Class frequencies and weights from the TRAINING patches only
counts = np.bincount(CACHE["train"]["lbl"].ravel(), minlength=256)[:NUM_CLASSES].astype(np.float64)
freq = counts / counts.sum()
w = 1.0 / np.sqrt(freq + 1e-6)
w = np.clip(w / w.mean(), 0.5, 5.0)
CLASS_WEIGHTS = torch.tensor(w, dtype=torch.float32)
display(pd.DataFrame({"class": CLASS_NAMES, "train_pixel_share_%": (100 * freq).round(2),
                      "CE_weight": w.round(3)}))

# %%
# ============================================================
# CELL 4 — DATASETS AND LOADERS (in-memory, no TIFF reads)
# ============================================================
class CachedPatches(Dataset):
    def __init__(self, d, augment=False):
        self.o, self.s, self.y = d["opt"], d["sar"], d["lbl"]
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        o = torch.from_numpy(self.o[i]).float().div_(255.0)
        s = torch.from_numpy(self.s[i]).float().div_(255.0)
        y = torch.from_numpy(self.y[i].astype(np.int64))
        if self.augment:
            if random.random() < 0.5:
                o, s, y = o.flip(-1), s.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                o, s, y = o.flip(-2), s.flip(-2), y.flip(-2)
            k = random.randint(0, 3)
            if k:
                o, s, y = (torch.rot90(o, k, (-2, -1)), torch.rot90(s, k, (-2, -1)),
                           torch.rot90(y, k, (-2, -1)))
        return o.contiguous(), s.contiguous(), y.contiguous()

train_ds = CachedPatches(CACHE["train"], augment=True)
val_ds = CachedPatches(CACHE["val"])
test_ds = CachedPatches(CACHE["test"])

def make_train_loader(seed):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                      num_workers=0, pin_memory=False, generator=g)

val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} patches "
      f"| {len(train_ds) // BATCH_SIZE} train batches/epoch")

# %%
# ============================================================
# CELL 5 — MODEL (GroupNorm), LOSSES, CIMD FUNCTIONS, METRICS
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(GN_GROUPS, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(GN_GROUPS, cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, cin, base=24):
        super().__init__()
        self.b1, self.b2, self.b3 = ConvBlock(cin, base), ConvBlock(base, base * 2), ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.b1(x); f2 = self.b2(self.pool(f1)); f3 = self.b3(self.pool(f2))
        return f1, f2, f3

class CIMDNet(nn.Module):
    def __init__(self, opt_ch, sar_ch, ncls=7, base=24):
        super().__init__()
        self.opt_encoder = Encoder(opt_ch, base)
        self.sar_encoder = Encoder(sar_ch, base)
        self.fuse3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.seg_head = nn.Conv2d(base, ncls, 1)
        self.dep_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(base * 8, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, ncls * 2))
        self.ncls = ncls

    def forward(self, optical, sar, mode="full", return_dependency=False):
        if mode == "opt":
            sar = torch.zeros_like(sar)
        elif mode == "sar":
            optical = torch.zeros_like(optical)
        elif mode != "full":
            raise ValueError(mode)
        o1, o2, o3 = self.opt_encoder(optical)
        s1, s2, s3 = self.sar_encoder(sar)
        z = self.fuse3(torch.cat([o3, s3], 1))
        z = self.dec2(torch.cat([self.up2(z), o2, s2], 1))
        z = self.dec1(torch.cat([self.up1(z), o1, s1], 1))
        logits = self.seg_head(z)
        if return_dependency:
            dep = torch.sigmoid(self.dep_head(torch.cat([o3, s3], 1)).float()).view(-1, self.ncls, 2)
            return logits, dep
        return logits

def make_model():
    return CIMDNet(OPT_CH, SAR_CH, NUM_CLASSES).to(DEVICE)

print("Parameters: %.2f M" % (sum(p.numel() for p in make_model().parameters()) / 1e6))

# ---- segmentation loss ----
CE = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE) if USE_CLASS_WEIGHTS else None,
                         ignore_index=IGNORE_INDEX)

def dice_loss(logits, y, eps=1e-6):
    probs = torch.softmax(logits.float(), 1)
    valid = y != IGNORE_INDEX
    yy = y.clone(); yy[~valid] = 0
    onehot = F.one_hot(yy, NUM_CLASSES).permute(0, 3, 1, 2).float()
    m = valid.unsqueeze(1).float()
    probs, onehot = probs * m, onehot * m
    inter = (probs * onehot).sum((0, 2, 3))
    den = (probs + onehot).sum((0, 2, 3))
    dice = (2 * inter + eps) / (den + eps)
    present = onehot.sum((0, 2, 3)) > 0
    return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0

def segmentation_loss(logits, y):
    return CE(logits.float(), y) + 0.5 * dice_loss(logits, y)

# ---- CIMD: interventional class-conditional dependency (unchanged from v1) ----
def intervention_dependency(p_full, p_missing, y, tau=TAU):
    ds, vs = [], []
    for c in range(NUM_CLASSES):
        mask = (y == c).float()
        cnt = mask.sum((1, 2))
        r = (torch.abs(p_full[:, c] - p_missing[:, c]) * mask).sum((1, 2)) / (cnt + 1e-8)
        ds.append(r / (r + tau)); vs.append(cnt > 0)
    return torch.stack(ds, 1), torch.stack(vs, 1)

def masked_dependency_loss(d_pred, d_true, valid):
    v = valid.float()
    return (torch.abs(d_pred - d_true.detach()) * v).sum() / (v.sum() + 1e-8)

def uniform_consistency(p_full, p_other, y):
    valid = (y != IGNORE_INDEX).unsqueeze(1).float()
    return (torch.abs(p_full - p_other) * valid).sum() / (valid.sum() * p_full.shape[1] + 1e-8)

def weighted_class_response_loss(response, weight, valid):
    v = valid.float()
    return (response * weight * v).sum() / (v.sum() + 1e-8)

def compute_loss(name, model, optical, sar, y):
    if name == "A0":
        return segmentation_loss(model(optical, sar, "full"), y)

    if name == "A3":
        logits_full, d_pred = model(optical, sar, "full", return_dependency=True)
    else:
        logits_full = model(optical, sar, "full")
    logits_opt = model(optical, sar, "opt")
    logits_sar = model(optical, sar, "sar")
    loss = segmentation_loss(logits_full, y) + ALPHA * (
        segmentation_loss(logits_opt, y) + segmentation_loss(logits_sar, y))
    if name == "A1":
        return loss

    p_full = torch.softmax(logits_full.float(), 1)
    p_opt = torch.softmax(logits_opt.float(), 1)
    p_sar = torch.softmax(logits_sar.float(), 1)

    if name == "A2":
        cons = 0.5 * (uniform_consistency(p_full, p_opt, y) + uniform_consistency(p_full, p_sar, y))
        return loss + GAMMA * cons

    if name == "A3":
        d_opt, v_opt = intervention_dependency(p_full, p_sar, y)   # optical removed -> SAR-only
        d_sar, v_sar = intervention_dependency(p_full, p_opt, y)   # SAR removed -> optical-only
        d_true = torch.stack([d_opt, d_sar], 2)
        valid = torch.stack([v_opt, v_sar], 2)
        l_dep = masked_dependency_loss(d_pred, d_true, valid)
        r_opt = TAU * d_opt / (1.0 - d_opt + 1e-6)
        r_sar = TAU * d_sar / (1.0 - d_sar + 1e-6)
        response = torch.stack([r_opt, r_sar], 2)
        l_dwcr = weighted_class_response_loss(response, 1.0 - d_pred.detach(), valid)
        l_comp = weighted_class_response_loss(response, d_pred.detach(), valid)
        return loss + BETA * l_dep + GAMMA * l_dwcr + DELTA * l_comp

    raise ValueError(name)

# ---- metrics ----
def metrics_from_confusion(cm):
    cm = cm.numpy().astype(np.float64)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12); rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"mIoU": float(iou.mean()), "macroF1": float(f1.mean()),
            "OA": float(tp.sum() / (cm.sum() + 1e-12)),
            "per_class_iou": iou.tolist(), "per_class_f1": f1.tolist()}

@torch.no_grad()
def evaluate(model, loader, mode="full", degrade=None, desc=None):
    """degrade: optional function (optical, sar) -> (optical, sar) applied before a FULL-mode pass."""
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for bi, (o, s, y) in enumerate(tqdm(loader, leave=False, desc=desc or f"eval-{mode}")):
        o, s, y = o.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
        if degrade is not None:
            o, s = degrade(o, s, bi)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            pred = model(o, s, mode=mode).argmax(1)
        m = y != IGNORE_INDEX
        cm += torch.bincount(y[m] * NUM_CLASSES + pred[m],
                             minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).cpu()
    return metrics_from_confusion(cm), cm

def load_checkpoint(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = make_model()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck

print("Setup complete.")

# %%
# ============================================================
# CELL 6 — RESUME-SAFE TRAINING FUNCTION
# ============================================================
def train_experiment(name, seed=42):
    tag = f"{name}_s{seed}"
    best_path = CKPT_DIR / f"{tag}_best.pt"
    latest_path = CKPT_DIR / f"{tag}_latest.pt"
    done_path = CKPT_DIR / f"{tag}_DONE.json"
    if done_path.exists() and best_path.exists():
        print(f"{tag} already finished -> {best_path.name} (delete {done_path.name} to retrain)")
        return best_path

    set_seed(seed)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    start_epoch, best_score, bad_epochs, history = 1, -1.0, 0, []

    if latest_path.exists():
        st = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
        start_epoch, best_score = st["epoch"] + 1, st["best_score"]
        bad_epochs, history = st["bad_epochs"], st["history"]
        print(f"Resuming {tag} from epoch {start_epoch} (best robust score so far {best_score:.4f})")

    train_loader = make_train_loader(seed)
    print("=" * 70, f"\nTRAINING {tag} — {EXPERIMENTS[name]}\n" + "=" * 70)

    for epoch in range(start_epoch, EPOCHS + 1):
        if bad_epochs >= PATIENCE:
            break
        model.train()
        t0, total = time.time(), 0.0
        for o, s, y in tqdm(train_loader, leave=False, desc=f"{tag} ep{epoch:02d}"):
            o, s, y = o.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                loss = compute_loss(name, model, o, s, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.item())
        scheduler.step()

        vf, _ = evaluate(model, val_loader, "full")
        vo, _ = evaluate(model, val_loader, "opt")
        vs, _ = evaluate(model, val_loader, "sar")
        robust = float(np.mean([vf["mIoU"], vo["mIoU"], vs["mIoU"]]))
        row = {"experiment": name, "seed": seed, "epoch": epoch,
               "train_loss": total / max(1, len(train_loader)),
               "val_full_mIoU": vf["mIoU"], "val_opt_mIoU": vo["mIoU"], "val_sar_mIoU": vs["mIoU"],
               "val_robust": robust, "minutes": (time.time() - t0) / 60}
        history.append(row)

        improved = robust > best_score
        if improved:
            best_score, bad_epochs = robust, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "robust_score": robust,
                        "experiment": name, "seed": seed, "opt_channels": OPT_CH,
                        "sar_channels": SAR_CH, "norm": "group"}, best_path)
        else:
            bad_epochs += 1

        print(f"{tag} ep{epoch:02d} | loss {row['train_loss']:.3f} | val FULL {vf['mIoU']:.4f} "
              f"OPT {vo['mIoU']:.4f} SAR {vs['mIoU']:.4f} | robust {robust:.4f}"
              f"{' *best*' if improved else ''} | {row['minutes']:.1f} min")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_score": best_score, "bad_epochs": bad_epochs,
                    "history": history}, latest_path)
        pd.DataFrame(history).to_csv(OUT_DIR / f"{tag}_history.csv", index=False)

    if bad_epochs >= PATIENCE:
        print(f"Early stopping: no improvement for {PATIENCE} epochs.")
    done_path.write_text(json.dumps({"best_robust": best_score, "epochs_run": len(history)}))
    print(f"{tag} finished. Best validation robust score {best_score:.4f}")
    return best_path

# %%
# ============================================================
# CELL 7 — TRAIN A0 (full-modality baseline)
# ============================================================
for s in SEEDS:
    train_experiment("A0", s)

# %%
# ============================================================
# CELL 8 — TRAIN A1 (missing-modality supervised)
# ============================================================
for s in SEEDS:
    train_experiment("A1", s)

# %%
# ============================================================
# CELL 9 — TRAIN A2 (uniform consistency) and A3 (CIMD + DWCR)
# Split into two cells if you prefer; each resumes independently.
# ============================================================
for s in SEEDS:
    train_experiment("A2", s)
for s in SEEDS:
    train_experiment("A3", s)

# %%
# ============================================================
# CELL 10 — TEST: FULL / OPTICAL-ONLY / SAR-ONLY  (needs Cells 1–5 only)
# ============================================================
ckpts = sorted(CKPT_DIR.glob("A?_s*_best.pt")) + sorted(CKPT_DIR.glob("A0F_s*_best.pt"))
if not ckpts:
    raise RuntimeError("No trained checkpoints in " + str(CKPT_DIR))

summary_rows, per_class_rows = [], []
for path in ckpts:
    model, ck = load_checkpoint(path)
    exp, seed = ck["experiment"], ck["seed"]
    res = {}
    for mode in ["full", "opt", "sar"]:
        m, cm = evaluate(model, test_loader, mode, desc=f"{exp}_s{seed}-{mode}")
        res[mode] = m
        pd.DataFrame(cm.numpy(), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            OUT_DIR / f"{exp}_s{seed}_{mode}_confusion.csv")
        for c, cname in enumerate(CLASS_NAMES):
            per_class_rows.append({"experiment": exp, "seed": seed, "mode": mode, "class": cname,
                                   "IoU": m["per_class_iou"][c], "F1": m["per_class_f1"][c]})
    full, optm, sarm = res["full"]["mIoU"], res["opt"]["mIoU"], res["sar"]["mIoU"]
    summary_rows.append({
        "experiment": exp, "seed": seed, "best_epoch": ck["epoch"],
        "FULL_mIoU": full, "OPT_mIoU": optm, "SAR_mIoU": sarm,
        "FULL_F1": res["full"]["macroF1"], "OPT_F1": res["opt"]["macroF1"], "SAR_F1": res["sar"]["macroF1"],
        "FULL_OA": res["full"]["OA"], "OPT_OA": res["opt"]["OA"], "SAR_OA": res["sar"]["OA"],
        "PRR_opt_only": optm / (full + 1e-12), "PRR_sar_only": sarm / (full + 1e-12),
        "Robust_mean_mIoU": float(np.mean([full, optm, sarm])),
        "Worst_case_mIoU": float(min(full, optm, sarm))})
    print(f"{exp}_s{seed}: FULL {full:.4f} | OPT-only {optm:.4f} | SAR-only {sarm:.4f}")

results_df = pd.DataFrame(summary_rows)
perclass_df = pd.DataFrame(per_class_rows)
results_df.to_csv(OUT_DIR / "test_results_per_seed.csv", index=False)
perclass_df.to_csv(OUT_DIR / "test_per_class_per_seed.csv", index=False)

metric_cols = [c for c in results_df.columns if c not in ("experiment", "seed", "best_epoch")]
agg = results_df.groupby("experiment")[metric_cols].agg(["mean", "std"])
agg.to_csv(OUT_DIR / "test_results_mean_std.csv")

print("\nMean over seeds (std shown when more than one seed was run):")
main = results_df.groupby("experiment")[["FULL_mIoU", "OPT_mIoU", "SAR_mIoU", "Robust_mean_mIoU",
                                         "Worst_case_mIoU", "PRR_opt_only", "PRR_sar_only"]].mean()
display(main.round(4))
display(perclass_df.groupby(["class", "experiment", "mode"])["IoU"].mean()
        .unstack(["experiment", "mode"]).round(3))

# %%
# ============================================================
# CELL 11 — A3 DEPENDENCY ANALYSIS (predicted vs interventional)
# ============================================================
a3_ckpts = sorted(CKPT_DIR.glob("A3_s*_best.pt"))
if not a3_ckpts:
    print("A3 not trained yet — skipping.")
else:
    rows = []
    for path in a3_ckpts:
        model, ck = load_checkpoint(path)
        with torch.no_grad():
            for o, s, y in tqdm(test_loader, desc=f"dependency {path.stem}"):
                o, s, y = o.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
                lf, d_pred = model(o, s, "full", return_dependency=True)
                pf = torch.softmax(lf.float(), 1)
                po = torch.softmax(model(o, s, "opt").float(), 1)
                ps = torch.softmax(model(o, s, "sar").float(), 1)
                d_opt, v_opt = intervention_dependency(pf, ps, y)
                d_sar, v_sar = intervention_dependency(pf, po, y)
                for mi, (dt, vv) in enumerate([(d_opt, v_opt), (d_sar, v_sar)]):
                    b_idx, c_idx = torch.nonzero(vv, as_tuple=True)
                    for b, c in zip(b_idx.tolist(), c_idx.tolist()):
                        rows.append({"seed": ck["seed"], "class": CLASS_NAMES[c],
                                     "modality": ["optical", "sar"][mi],
                                     "D_true": float(dt[b, c]), "D_pred": float(d_pred[b, c, mi])})
    dep_df = pd.DataFrame(rows)
    dep_df["abs_err"] = (dep_df["D_true"] - dep_df["D_pred"]).abs()
    dep_summary = (dep_df.groupby(["class", "modality"])
                   .agg(D_true=("D_true", "mean"), D_pred=("D_pred", "mean"),
                        MAE=("abs_err", "mean"), n=("D_true", "size")).reset_index())
    corr = pd.Series({mod: g["D_true"].corr(g["D_pred"], method="spearman")
                      for mod, g in dep_df.groupby("modality")})
    dep_df.to_csv(OUT_DIR / "A3_dependency_samples.csv", index=False)
    dep_summary.to_csv(OUT_DIR / "A3_dependency_summary.csv", index=False)
    display(dep_summary.round(4))
    print("Spearman correlation (predicted vs interventional dependency):")
    print(corr.round(3).to_string())

    piv = dep_summary.pivot(index="class", columns="modality", values="D_true").reindex(CLASS_NAMES)
    ax = piv.plot.bar(figsize=(8, 4), rot=0)
    ax.set_ylabel("Interventional dependency D"); ax.set_title("Class–modality dependency (A3, test)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "A3_class_dependency.png", dpi=200); plt.show()

# %%
# ============================================================
# CELL 12 — PROGRESSIVE SENSOR DEGRADATION (Objective 4)
# A fraction p of 32×32 blocks of one modality is blanked (cloud gaps for
# optical, data gaps for SAR). p = 0 is the full input; p = 1 equals the
# fully missing modality. The same block masks are used for every model.
# ============================================================
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
BLOCK = 32

def make_degrader(which, p, seed=2026):
    def degrade(o, s, batch_idx):
        g = torch.Generator(device="cpu").manual_seed(seed + batch_idx)
        B, _, H, W = o.shape
        keep = (torch.rand(B, 1, H // BLOCK, W // BLOCK, generator=g) >= p).float()
        keep = F.interpolate(keep, size=(H, W), mode="nearest").to(o.device)
        return (o * keep, s) if which == "optical" else (o, s * keep)
    return degrade

deg_rows = []
for path in sorted(CKPT_DIR.glob("A?_s*_best.pt")) + sorted(CKPT_DIR.glob("A0F_s*_best.pt")):
    model, ck = load_checkpoint(path)
    for which in ["optical", "sar"]:
        for p in LEVELS:
            m, _ = evaluate(model, test_loader, "full", degrade=make_degrader(which, p),
                            desc=f"{path.stem} {which} p={p}")
            deg_rows.append({"experiment": ck["experiment"], "seed": ck["seed"],
                             "degraded": which, "level": p, "mIoU": m["mIoU"]})

deg_df = pd.DataFrame(deg_rows)
deg_df.to_csv(OUT_DIR / "degradation_results.csv", index=False)
deg_mean = deg_df.groupby(["degraded", "experiment", "level"])["mIoU"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, which in zip(axes, ["optical", "sar"]):
    for exp in sorted(deg_mean["experiment"].unique()):
        d = deg_mean[(deg_mean["degraded"] == which) & (deg_mean["experiment"] == exp)]
        ax.plot(d["level"], d["mIoU"], marker="o", label=exp)
    ax.set_title(f"{which.upper()} progressively removed")
    ax.set_xlabel("Fraction of blocks removed"); ax.grid(alpha=0.3)
axes[0].set_ylabel("Test mIoU"); axes[1].legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "degradation_curves.png", dpi=200); plt.show()

# Area under the degradation curve (higher = more robust)
def _area(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))

auc = pd.DataFrame([{"experiment": e, f"AUC_{w}_degraded": _area(d["level"], d["mIoU"])}
                    for (w, e), d in deg_mean.groupby(["degraded", "experiment"])])
auc = auc.groupby("experiment").first().round(4)
auc.to_csv(OUT_DIR / "degradation_auc.csv")
display(auc)

# %%
# ============================================================
# CELL 13 — SUMMARY FIGURE + PACKAGE ALL RESULTS
# ============================================================
if "results_df" not in globals():
    results_df = pd.read_csv(OUT_DIR / "test_results_per_seed.csv")
plot_df = results_df.groupby("experiment")[["FULL_mIoU", "OPT_mIoU", "SAR_mIoU"]].mean()
ax = plot_df.plot.bar(figsize=(8, 4), rot=0)
ax.set_ylabel("Test mIoU"); ax.set_title("A0–A3 under full, optical-only and SAR-only input")
ax.legend(["Full", "Optical only", "SAR only"]); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DIR / "A0_A3_modality_comparison.png", dpi=200); plt.show()

zip_path = shutil.make_archive(str(ROOT / "CIMD_GeoFM_GN_results"), "zip", OUT_DIR)
print("All results packaged:", zip_path)

# %% [markdown]
# # Part 2 — Fair baseline, failure forecast and ablations
#
# Run Cells 1–5 first. **A0F** (Cell 21) is the full-modality baseline reported in the paper (selected on validation FULL mIoU, no early stopping); the robust-selected A0 above is kept only for completeness. Cells 14–17 and 19 reproduce the negative / ablation results (loss-term magnitudes, DWCR strength sweep, one-way distillation, damage-target estimator).

# %%
# ============================================================
# CELL 21 — FAIR A0 BASELINE (A0F): select on FULL accuracy, no early stop
# Run after Cells 1–5.  ~8 min per seed (~25 min for 3 seeds).
#
# Why: A0 never sees a missing modality, so its OPT-only/SAR-only scores
# are near zero and jump around. Selecting A0 on the robust score
# (mean of FULL/OPT/SAR) made it stop early (epoch 3–7 in seeds 43/44)
# and under-reports its FULL accuracy. A reviewer would object.
# A0F is the standard full-modality baseline: trained 25 epochs and
# selected on validation FULL mIoU. Saved separately as A0F_s*_best.pt.
# ============================================================
A0F_SEEDS = [42, 43, 44]

def train_a0f(seed):
    best_path = CKPT_DIR / f"A0F_s{seed}_best.pt"
    if best_path.exists():
        print(f"A0F_s{seed} already trained -> {best_path.name}")
        return best_path
    set_seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    loader = make_train_loader(seed)
    best, hist = -1.0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for o, s, y in tqdm(loader, leave=False, desc=f"A0F_s{seed} ep{epoch:02d}", mininterval=10):
            o, s, y = o.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                loss = segmentation_loss(model(o, s, "full"), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        sched.step()
        vf, _ = evaluate(model, val_loader, "full")
        hist.append({"epoch": epoch, "val_full_mIoU": vf["mIoU"]})
        flag = ""
        if vf["mIoU"] > best:
            best, flag = vf["mIoU"], " *best*"
            torch.save({"model": model.state_dict(), "epoch": epoch, "robust_score": vf["mIoU"],
                        "experiment": "A0F", "seed": seed, "opt_channels": OPT_CH,
                        "sar_channels": SAR_CH, "norm": "group"}, best_path)
        print(f"A0F_s{seed} ep{epoch:02d} | val FULL {vf['mIoU']:.4f}{flag}")
    pd.DataFrame(hist).to_csv(OUT_DIR / f"A0F_s{seed}_history.csv", index=False)
    return best_path

for s in A0F_SEEDS:
    train_a0f(s)

# ---- test A0F under FULL / OPT-only / SAR-only and compare with A0, A1, A3 ----
rows = []
for path in sorted(CKPT_DIR.glob("A0F_s*_best.pt")):
    model, ck = load_checkpoint(path)
    r = {"experiment": "A0F", "seed": ck["seed"], "best_epoch": ck["epoch"]}
    for mode in ["full", "opt", "sar"]:
        r[f"{mode.upper()}_mIoU"] = evaluate(model, test_loader, mode)[0]["mIoU"]
    rows.append(r)
a0f_df = pd.DataFrame(rows)
a0f_df.to_csv(OUT_DIR / "A0F_test_results_per_seed.csv", index=False)

prev = pd.read_csv(OUT_DIR / "test_results_per_seed.csv")
allr = pd.concat([prev[["experiment", "seed", "best_epoch", "FULL_mIoU", "OPT_mIoU", "SAR_mIoU"]], a0f_df])
allr["Robust_mean"] = allr[["FULL_mIoU", "OPT_mIoU", "SAR_mIoU"]].mean(axis=1)
allr.to_csv(OUT_DIR / "test_results_per_seed_with_A0F.csv", index=False)

table = allr.groupby("experiment")[["FULL_mIoU", "OPT_mIoU", "SAR_mIoU", "Robust_mean"]].agg(["mean", "std"])
table.to_csv(OUT_DIR / "test_results_mean_std_with_A0F.csv")
print("\nPer seed:")
display(allr.sort_values(["experiment", "seed"]).round(4))
print("Mean ± std over seeds:")
display(table.round(4))

# %%
# ============================================================
# CELL 18 — DOES PREDICTED DEPENDENCY FORECAST SENSOR-FAILURE DAMAGE?
# Run after Cells 1–5. No training, ~2–3 min per A3 checkpoint.
#
# Question for the paper: from ONE full-input forward pass, can D_pred
# tell us which classes (and which image patches) will lose accuracy if
# the optical or the SAR sensor fails?
#   removing OPTICAL -> SAR-only prediction  <-> D_pred[..., 0]
#   removing SAR     -> optical-only prediction <-> D_pred[..., 1]
# ============================================================
FORECAST_GLOB = "A3_s*_best.pt"   # "A3D_s*_best.pt" to test the damage estimator (Cell 19)
MIN_PIXELS = 256          # ignore class instances smaller than this in a patch (too noisy)
MIN_IOU_FULL = 0.10       # class-level tests skip classes the model barely segments (e.g. "others")
FAIL_THRESHOLD = 0.20     # "failure" = class recall drops by more than 20 points

def auroc(scores, labels):
    """Probability that a failing instance gets a higher score than a non-failing one."""
    labels = np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(np.asarray(scores)).rank().values
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

a3_ckpts = sorted(CKPT_DIR.glob(FORECAST_GLOB))
if not a3_ckpts:
    raise RuntimeError(f"No {FORECAST_GLOB} checkpoint found in {CKPT_DIR}")

inst_rows, class_rows = [], []
for path in a3_ckpts:
    model, ck = load_checkpoint(path)
    seed = ck["seed"]
    cms = {m: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64) for m in ["full", "opt", "sar"]}
    dsum = torch.zeros(NUM_CLASSES, 2); dcnt = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for o, s, y in tqdm(test_loader, desc=f"failure forecast {path.stem}"):
            o, s, y = o.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
            lf, d_pred = model(o, s, "full", return_dependency=True)
            preds = {"full": lf.argmax(1), "opt": model(o, s, "opt").argmax(1), "sar": model(o, s, "sar").argmax(1)}
            valid = y != IGNORE_INDEX
            for m, p in preds.items():
                cms[m] += torch.bincount(y[valid] * NUM_CLASSES + p[valid],
                                         minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).cpu()
            d_cpu = d_pred.float().cpu()
            for c in range(NUM_CLASSES):
                mask = (y == c)
                cnt = mask.sum((1, 2))
                present = torch.nonzero(cnt > 0).flatten().tolist()
                for b in present:
                    dsum[c] += d_cpu[b, c]; dcnt[c] += 1
                    n = int(cnt[b])
                    if n < MIN_PIXELS:
                        continue
                    rec = {m: float(((p[b] == c) & mask[b]).sum()) / n for m, p in preds.items()}
                    inst_rows.append({"seed": seed, "class": CLASS_NAMES[c], "removed": "optical",
                                      "D_pred": float(d_cpu[b, c, 0]), "recall_drop": rec["full"] - rec["sar"]})
                    inst_rows.append({"seed": seed, "class": CLASS_NAMES[c], "removed": "sar",
                                      "D_pred": float(d_cpu[b, c, 1]), "recall_drop": rec["full"] - rec["opt"]})

    iou = {m: np.array(metrics_from_confusion(cm)["per_class_iou"]) for m, cm in cms.items()}
    for c, cname in enumerate(CLASS_NAMES):
        for mi, removed, kept in [(0, "optical", "sar"), (1, "sar", "opt")]:
            class_rows.append({"seed": seed, "class": cname, "removed": removed,
                               "D_pred_mean": float(dsum[c, mi] / max(dcnt[c], 1)),
                               "IoU_full": iou["full"][c], "IoU_after_removal": iou[kept][c],
                               "IoU_drop": iou["full"][c] - iou[kept][c]})

inst_df = pd.DataFrame(inst_rows)
class_df = (pd.DataFrame(class_rows)
            .groupby(["class", "removed"], as_index=False)
            [["D_pred_mean", "IoU_full", "IoU_after_removal", "IoU_drop"]].mean())
inst_df.to_csv(OUT_DIR / f"failure_forecast_instances_{FORECAST_GLOB.split('_')[0]}.csv", index=False)
class_df.to_csv(OUT_DIR / f"failure_forecast_classes_{FORECAST_GLOB.split('_')[0]}.csv", index=False)

# ---- 1. Class level: does D_pred rank classes by how much they lose? ----
cls_eval = class_df[class_df["IoU_full"] >= MIN_IOU_FULL]
print("Class-level tests use:", sorted(cls_eval["class"].unique()))
cls_rho = cls_eval["D_pred_mean"].corr(cls_eval["IoU_drop"], method="spearman")
cls_rho_by_mod = {m: g["D_pred_mean"].corr(g["IoU_drop"], method="spearman")
                  for m, g in cls_eval.groupby("removed")}

# ---- 2. Instance level: does D_pred rank individual patches? ----
inst_rho = inst_df["D_pred"].corr(inst_df["recall_drop"], method="spearman")
within = [g["D_pred"].corr(g["recall_drop"], method="spearman")
          for _, g in inst_df.groupby(["class", "removed"]) if len(g) >= 20]
within_rho = float(np.nanmean(within)) if within else np.nan

# ---- 3. Failure detection: can D_pred flag instances that will fail? ----
fail = inst_df["recall_drop"] > FAIL_THRESHOLD
auc_all = auroc(inst_df["D_pred"], fail)
auc_by_mod = {m: auroc(g["D_pred"], g["recall_drop"] > FAIL_THRESHOLD) for m, g in inst_df.groupby("removed")}

# ---- 4. Does D_pred beat knowing only the class? ----
# class-avg D  : replace each instance's D_pred with its class mean (what a class lookup table gives)
# oracle prior : each class's TRUE mean damage on the test set (an upper bound for any class-only rule)
base_rows = []
for m, g in inst_df.groupby("removed"):
    f = g["recall_drop"] > FAIL_THRESHOLD
    base_rows.append({"removed": m, "fail_rate": f.mean(),
                      "AUROC_D_pred": auroc(g["D_pred"], f),
                      "AUROC_class_avg_D": auroc(g.groupby("class")["D_pred"].transform("mean"), f),
                      "AUROC_oracle_class_prior": auroc(g.groupby("class")["recall_drop"].transform("mean"), f)})
baseline_df = pd.DataFrame(base_rows)
baseline_df.to_csv(OUT_DIR / f"failure_forecast_vs_class_prior_{FORECAST_GLOB.split('_')[0]}.csv", index=False)

summary = pd.DataFrame([
    {"test": f"Class level: D_pred vs IoU drop (Spearman, {len(cls_eval)} points)", "value": cls_rho},
    {"test": "   optical removed", "value": cls_rho_by_mod.get("optical")},
    {"test": "   SAR removed", "value": cls_rho_by_mod.get("sar")},
    {"test": f"Instance level: D_pred vs recall drop (Spearman, n={len(inst_df)})", "value": inst_rho},
    {"test": "   mean WITHIN-class Spearman (beyond class identity)", "value": within_rho},
    {"test": f"Failure detection AUROC (recall drop > {FAIL_THRESHOLD:.2f}), failure rate {fail.mean():.1%}", "value": auc_all},
    {"test": "   optical removed", "value": auc_by_mod.get("optical")},
    {"test": "   SAR removed", "value": auc_by_mod.get("sar")},
])
tagname = FORECAST_GLOB.split("_")[0]
summary.to_csv(OUT_DIR / f"failure_forecast_summary_{tagname}.csv", index=False)
display(class_df.round(4))
display(summary.round(3))
print("D_pred vs class-only baselines (AUROC):")
display(baseline_df.round(3))

fig, ax = plt.subplots(figsize=(6.5, 5))
for removed, marker in [("optical", "o"), ("sar", "s")]:
    d = class_df[class_df["removed"] == removed]
    ax.scatter(d["D_pred_mean"], d["IoU_drop"], marker=marker, s=60, label=f"{removed.upper()} removed")
    for _, r in d.iterrows():
        ax.annotate(r["class"], (r["D_pred_mean"], r["IoU_drop"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("Predicted dependency D_pred (full-input pass)")
ax.set_ylabel("Actual IoU drop after sensor removal")
ax.set_title(f"Failure forecast — class-level Spearman {cls_rho:.2f}")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / f"failure_forecast_scatter_{FORECAST_GLOB.split('_')[0]}.png", dpi=200); plt.show()

print("How to read this:")
print("  Within-class Spearman > ~0.3 and AUROC > ~0.70 -> D_pred is a usable failure forecast (strong paper hook).")
print("  High class-level but ~0 within-class -> it only knows class averages (still reportable, weaker).")
print("  Near 0 everywhere -> D_pred describes dependency but does not forecast damage; rethink the hook.")

# %%
# ============================================================
# CELL 14 — LOSS-TERM MAGNITUDE DIAGNOSTIC (no training, ~1 min)
# Run after Cells 1–5. Measures how large each A3 loss term is
# compared with the segmentation loss, using the trained A3 model.
# ============================================================
@torch.no_grad()
def loss_term_magnitudes(ckpt_path, n_batches=40):
    model, ck = load_checkpoint(ckpt_path)
    g = torch.Generator(); g.manual_seed(0)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0)
    acc = defaultdict(list)
    for i, (o, s, y) in enumerate(loader):
        if i >= n_batches:
            break
        o, s, y = o.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
        lf, d_pred = model(o, s, "full", return_dependency=True)
        lo, ls = model(o, s, "opt"), model(o, s, "sar")
        pf, po, ps = (torch.softmax(t.float(), 1) for t in (lf, lo, ls))

        d_opt, v_opt = intervention_dependency(pf, ps, y)
        d_sar, v_sar = intervention_dependency(pf, po, y)
        d_true = torch.stack([d_opt, d_sar], 2)
        valid = torch.stack([v_opt, v_sar], 2)
        r = torch.stack([TAU * d_opt / (1 - d_opt + 1e-6), TAU * d_sar / (1 - d_sar + 1e-6)], 2)

        acc["seg_full"].append(segmentation_loss(lf, y).item())
        acc["seg_missing (x ALPHA)"].append(ALPHA * (segmentation_loss(lo, y) + segmentation_loss(ls, y)).item())
        acc["dependency (x BETA)"].append(BETA * masked_dependency_loss(d_pred, d_true, valid).item())
        acc["DWCR (x GAMMA)"].append(GAMMA * weighted_class_response_loss(r, 1 - d_pred, valid).item())
        acc["compensation (x DELTA)"].append(DELTA * weighted_class_response_loss(r, d_pred, valid).item())
        acc["A2 consistency (x GAMMA)"].append(GAMMA * 0.5 * (uniform_consistency(pf, po, y)
                                                               + uniform_consistency(pf, ps, y)).item())
    df = pd.DataFrame({k: [np.mean(v)] for k, v in acc.items()}).T.rename(columns={0: "mean_value"})
    seg_total = df.loc["seg_full", "mean_value"] + df.loc["seg_missing (x ALPHA)", "mean_value"]
    df["percent_of_segmentation_loss"] = 100 * df["mean_value"] / seg_total
    return df.round(4)

a3_path = sorted(CKPT_DIR.glob("A3_s*_best.pt"))[0]
mag_df = loss_term_magnitudes(a3_path)
mag_df.to_csv(OUT_DIR / "A3_loss_term_magnitudes.csv")
display(mag_df)
print("If DWCR + compensation are only a few % of the segmentation loss, "
      "they cannot change training much -> run Cell 15.")

# %%
# ============================================================
# CELL 15 — REGULARISATION-STRENGTH SWEEP FOR A2 AND A3 (validation only)
# Both methods get the same weights, so the comparison stays fair.
# Each run takes about as long as one normal experiment (~20 min).
# Results are written to separate sub-folders and do NOT replace the
# main A0–A3 checkpoints. Select the weight on VALIDATION, never on test.
# ============================================================
SWEEP_GAMMAS = [1.0, 3.0]          # add 10.0 if 3.0 is still tied with A1
BASE = {"CKPT_DIR": CKPT_DIR, "OUT_DIR": OUT_DIR, "GAMMA": GAMMA, "DELTA": DELTA}
sweep_rows = []
try:
    for gval in SWEEP_GAMMAS:
        GAMMA, DELTA = gval, gval / 3.0            # keep the original 3:1 DWCR:compensation ratio
        OUT_DIR = BASE["OUT_DIR"] / f"sweep_gamma_{gval:g}"
        CKPT_DIR = OUT_DIR / "checkpoints"
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        for exp in ["A2", "A3"]:
            for s in SEEDS:
                train_experiment(exp, s)
                h = pd.read_csv(OUT_DIR / f"{exp}_s{s}_history.csv")
                b = h.loc[h["val_robust"].idxmax()]
                sweep_rows.append({"gamma": gval, "experiment": exp, "seed": s, "best_epoch": int(b["epoch"]),
                                   "val_FULL": b["val_full_mIoU"], "val_OPT": b["val_opt_mIoU"],
                                   "val_SAR": b["val_sar_mIoU"], "val_robust": b["val_robust"]})
finally:
    CKPT_DIR, OUT_DIR = BASE["CKPT_DIR"], BASE["OUT_DIR"]
    GAMMA, DELTA = BASE["GAMMA"], BASE["DELTA"]

# Reference: A1 and the original gamma = 0.3 runs (validation)
for exp in ["A1", "A2", "A3"]:
    for s in SEEDS:
        f = OUT_DIR / f"{exp}_s{s}_history.csv"
        if f.exists():
            h = pd.read_csv(f); b = h.loc[h["val_robust"].idxmax()]
            sweep_rows.append({"gamma": 0.0 if exp == "A1" else BASE["GAMMA"], "experiment": exp, "seed": s,
                               "best_epoch": int(b["epoch"]), "val_FULL": b["val_full_mIoU"],
                               "val_OPT": b["val_opt_mIoU"], "val_SAR": b["val_sar_mIoU"],
                               "val_robust": b["val_robust"]})

sweep_df = pd.DataFrame(sweep_rows).sort_values(["experiment", "gamma"])
sweep_df.to_csv(OUT_DIR / "gamma_sweep_validation.csv", index=False)
display(sweep_df.round(4))

# %%
# ============================================================
# CELL 16 — ONE-WAY (STOP-GRADIENT) DWCR  — redesigned regulariser
# Run after Cells 1–5.
#
# Why: the sweep showed that stronger consistency lowers FULL accuracy,
# because the gradient also pulls the full-modality prediction toward
# the degraded ones. Here the full-modality prediction is a fixed
# teacher (detached), so only the missing-modality predictions move.
#
#   A2KD : uniform one-way distillation, full -> missing (KD baseline)
#   A3KD : CIMD dependency-weighted one-way distillation (proposed v2)
#          weight = 1 - D_pred(class, removed modality), normalised so its
#          batch mean is 1. A3KD therefore has the SAME total strength as
#          A2KD; any difference comes only from the class-conditional
#          allocation, which is exactly what the paper has to show.
# ============================================================
EXPERIMENTS["A2KD"] = "Uniform one-way distillation (full -> missing)"
EXPERIMENTS["A3KD"] = "CIMD dependency-weighted one-way distillation"

if "_base_compute_loss" not in globals():
    _base_compute_loss = compute_loss          # keep A0–A3 behaviour unchanged

def classwise_kl(p_teacher, logits_student, y):
    """Per-sample, per-class mean of KL(teacher || student) over pixels of that class."""
    logp_s = F.log_softmax(logits_student.float(), 1)
    kl = (p_teacher * (torch.log(p_teacher + 1e-8) - logp_s)).sum(1)      # [B,H,W]
    vals, valid = [], []
    for c in range(NUM_CLASSES):
        m = (y == c).float()
        cnt = m.sum((1, 2))
        vals.append((kl * m).sum((1, 2)) / (cnt + 1e-8))
        valid.append(cnt > 0)
    return torch.stack(vals, 1), torch.stack(valid, 1)

def compute_loss(name, model, optical, sar, y):
    if name not in ("A2KD", "A3KD"):
        return _base_compute_loss(name, model, optical, sar, y)

    if name == "A3KD":
        logits_full, d_pred = model(optical, sar, "full", return_dependency=True)
    else:
        logits_full = model(optical, sar, "full")
    logits_opt = model(optical, sar, "opt")
    logits_sar = model(optical, sar, "sar")
    loss = segmentation_loss(logits_full, y) + ALPHA * (
        segmentation_loss(logits_opt, y) + segmentation_loss(logits_sar, y))

    p_teacher = torch.softmax(logits_full.float(), 1).detach()        # stop-gradient teacher
    kl_opt_removed, v1 = classwise_kl(p_teacher, logits_sar, y)       # optical removed -> SAR-only student
    kl_sar_removed, v2 = classwise_kl(p_teacher, logits_opt, y)       # SAR removed -> optical-only student
    kl = torch.stack([kl_opt_removed, kl_sar_removed], 2)             # [B,C,2]  (same order as D)
    valid = torch.stack([v1, v2], 2)

    if name == "A2KD":
        return loss + GAMMA * weighted_class_response_loss(kl, torch.ones_like(kl), valid)

    # A3KD: interventional dependency supervises the estimator (as before) ...
    p_opt = torch.softmax(logits_opt.float(), 1).detach()
    p_sar = torch.softmax(logits_sar.float(), 1).detach()
    d_opt, _ = intervention_dependency(p_teacher, p_sar, y)
    d_sar, _ = intervention_dependency(p_teacher, p_opt, y)
    l_dep = masked_dependency_loss(d_pred, torch.stack([d_opt, d_sar], 2), valid)

    # ... and the LEARNED dependency decides where invariance is enforced:
    # strong when the removed modality is redundant (low D), weak when it is essential.
    w = 1.0 - d_pred.detach()
    vf = valid.float()
    w = w / ((w * vf).sum() / (vf.sum() + 1e-8) + 1e-8)               # mean weight = 1
    return loss + BETA * l_dep + GAMMA * weighted_class_response_loss(kl, w, valid)

print("One-way DWCR registered: A2KD, A3KD")

# %%
# ============================================================
# CELL 17 — SWEEP A2KD vs A3KD (validation only)
# 2 strengths x 2 methods = 4 runs (~1.5 h). Sub-folders only;
# main A0–A3 checkpoints are not touched.
# ============================================================
SWEEP_GAMMAS = [0.5, 2.0]
BASE = {"CKPT_DIR": CKPT_DIR, "OUT_DIR": OUT_DIR, "GAMMA": GAMMA}
kd_rows = []

def _best_row(hist_csv, label, gval, exp, seed):
    h = pd.read_csv(hist_csv)
    b = h.loc[h["val_robust"].idxmax()]
    return {"setting": label, "gamma": gval, "experiment": exp, "seed": seed, "best_epoch": int(b["epoch"]),
            "val_FULL": b["val_full_mIoU"], "val_OPT": b["val_opt_mIoU"],
            "val_SAR": b["val_sar_mIoU"], "val_robust": b["val_robust"]}

try:
    for gval in SWEEP_GAMMAS:
        GAMMA = gval
        OUT_DIR = BASE["OUT_DIR"] / f"kd_sweep_gamma_{gval:g}"
        CKPT_DIR = OUT_DIR / "checkpoints"
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        for exp in ["A2KD", "A3KD"]:
            for s in SEEDS:
                train_experiment(exp, s)
                kd_rows.append(_best_row(OUT_DIR / f"{exp}_s{s}_history.csv", "one-way", gval, exp, s))
finally:
    CKPT_DIR, OUT_DIR, GAMMA = BASE["CKPT_DIR"], BASE["OUT_DIR"], BASE["GAMMA"]

# References from the main runs
for exp in ["A1", "A2", "A3"]:
    for s in SEEDS:
        f = OUT_DIR / f"{exp}_s{s}_history.csv"
        if f.exists():
            kd_rows.append(_best_row(f, "original", 0.0 if exp == "A1" else BASE["GAMMA"], exp, s))

kd_df = pd.DataFrame(kd_rows).sort_values(["setting", "experiment", "gamma"])
kd_df.to_csv(OUT_DIR / "one_way_dwcr_sweep_validation.csv", index=False)
display(kd_df.round(4))

# %%
# ============================================================
# CELL 19 — A3D: DAMAGE-TARGETED DEPENDENCY ESTIMATOR (~20 min)
# Run after Cells 1–5 (and 16 if you use A2KD/A3KD in the same session).
#
# Cell 18 showed that D_pred forecasts OPTICAL-removal damage well but
# misses SAR-removal damage (e.g. city: hurt most by losing SAR, yet
# rated least SAR-dependent). Reason: the original target measures the
# shift in class PROBABILITY. A low-confidence class such as city can
# flip its label with a small probability shift, so probability shift
# is not the same as damage.
#
# A3D keeps the same interventions (remove optical / remove SAR) but
# supervises the dependency head with the DECISION-level damage they
# cause: per-class recall drop, measured on the fly and detached.
# Segmentation losses are identical to A1; the DWCR terms (which did
# not help) are dropped, so any change is due to the estimator target.
# ============================================================
EXPERIMENTS["A3D"] = "Missing-modality training + interventional damage estimator"
DAMAGE_MIN_PIXELS = 64

if "_pre_a3d_compute_loss" not in globals():
    _pre_a3d_compute_loss = compute_loss

def damage_targets(pred_full, pred_missing, y, min_pixels=DAMAGE_MIN_PIXELS):
    """Per-sample, per-class recall drop caused by removing a modality, clipped to [0, 1]."""
    t, v = [], []
    for c in range(NUM_CLASSES):
        m = (y == c)
        cnt = m.sum((1, 2)).float()
        rf = ((pred_full == c) & m).sum((1, 2)).float() / cnt.clamp(min=1)
        rm = ((pred_missing == c) & m).sum((1, 2)).float() / cnt.clamp(min=1)
        t.append((rf - rm).clamp(0, 1)); v.append(cnt >= min_pixels)
    return torch.stack(t, 1), torch.stack(v, 1)

def compute_loss(name, model, optical, sar, y):
    if name != "A3D":
        return _pre_a3d_compute_loss(name, model, optical, sar, y)
    logits_full, d_pred = model(optical, sar, "full", return_dependency=True)
    logits_opt = model(optical, sar, "opt")
    logits_sar = model(optical, sar, "sar")
    loss = segmentation_loss(logits_full, y) + ALPHA * (
        segmentation_loss(logits_opt, y) + segmentation_loss(logits_sar, y))
    with torch.no_grad():
        pf, po, ps = logits_full.argmax(1), logits_opt.argmax(1), logits_sar.argmax(1)
        t_opt, v1 = damage_targets(pf, ps, y)      # optical removed -> SAR-only
        t_sar, v2 = damage_targets(pf, po, y)      # SAR removed -> optical-only
    target = torch.stack([t_opt, t_sar], 2)
    valid = torch.stack([v1, v2], 2)
    return loss + BETA * masked_dependency_loss(d_pred, target, valid)

for s in SEEDS:
    train_experiment("A3D", s)

print("\nNow run Cell 18 with FORECAST_GLOB = \"A3D_s*_best.pt\" and compare with the A3 result.")
