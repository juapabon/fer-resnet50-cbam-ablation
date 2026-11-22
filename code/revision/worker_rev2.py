"""Worker de los experimentos pedidos en el informe Q1 (cola rev2, con prioridades y reanudable).

Grupos, en orden de prioridad (protocolo y hiperparámetros del artículo salvo lo indicado):
  ft_ablacion    sin_cbam, sin_focal, sin_mixup y base con ajuste fino completo desde la época 1 (como completo_ft),
                 5 semillas                                                                  -> runs_ft_ablacion
  sin_balanceo   completo (focal) y sin_focal (entropía cruzada) sobre las imágenes originales, sin copias ni
                 muestreo ponderado; la pérdida usa pesos inversos a la frecuencia de clase  -> runs_sin_balanceo
  oficial_pi1    sin_cbam y las cuatro ubicaciones alternativas de CBAM en la partición oficial -> runs_oficial
  rafdb_color    completo y base en RAF-DB en color (balanceo con copias en color)          -> runs_rafdb_color
  convnext       completo y base con ConvNeXt-T preentrenada en ImageNet en lugar de ResNet-50 -> runs_convnext

Uso: python worker_rev2.py <id> [--prep | --pending]
"""
import copy, fcntl, json, os, random, sys, time, traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np

NB = "FER_ResNet50_CBAM_Frontiers.ipynb"
W = Path.home() / "fer_frontiers_work"
wid = sys.argv[1]
prep, pending_only = "--prep" in sys.argv, "--pending" in sys.argv
NUM_WORKERS = int(os.environ.get("FER_NUM_WORKERS", "5"))
SEEDS = [42, 43, 44, 45, 46]
GROUPS = [  # (grupo, carpeta, variantes)
    ("ft_ablacion", "runs_ft_ablacion", ["sin_cbam_ft", "sin_focal_ft", "sin_mixup_ft", "base_ft"]),
    ("sin_balanceo", "runs_sin_balanceo", ["completo", "sin_focal"]),
    ("oficial_pi1", "runs_oficial", ["sin_cbam", "cbam_layer2", "cbam_layer3", "cbam_layer3_4", "cbam_layer2_3_4"]),
    ("rafdb_color", "runs_rafdb_color", ["completo", "base"]),
    ("convnext", "runs_convnext", ["completo_cnx", "base_cnx"]),
]

def log(msg):
    print(f"{time.strftime('%F %T')} [rev2 w{wid}] {msg}", flush=True)

def done(runs, name, seed):
    return (W / runs / f"{name}_s{seed}" / "meta.json").exists()

def jobs():
    return [(grp, runs, v, s) for grp, runs, vs in GROUPS for s in SEEDS for v in vs if not done(runs, v, s)]

if pending_only:
    print(len(jobs()))
    sys.exit(0)

# ---------- entorno del cuaderno (celdas 0-22: datos, modelo, train_run) ----------
os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(Path(NB).parent)
cells = json.load(open(NB))["cells"]
g = {"__name__": "__main__"}
from IPython.display import display
g["display"] = display
for i, c in enumerate(cells[:23]):
    if c["cell_type"] == "code":
        exec(compile("".join(c["source"]), f"<celda {i}>", "exec"), g)
        if i == 4:
            g["CFG"]["num_workers"] = NUM_WORKERS
import torch, torch.nn as nn, pandas as pd
from PIL import Image
from torchvision import models, transforms
CFG, VARIANTS = g["CFG"], g["VARIANTS"]
CFG0 = copy.deepcopy(CFG)
WORK = g["WORK"]
exec(compile(open(W / "tmp" / "rev" / "train_run_reanudable.py").read(), "train_run_reanudable.py", "exec"), g)
exec(compile(open(W / "tmp" / "cells" / "c18_setup.py").read(), "c18_setup.py", "exec"), g)   # RAF-DB (gris)

@contextmanager
def data_context(**kw):
    old = {k: g[k] for k in kw}
    g.update(kw)
    try:
        yield
    finally:
        g.update(old)

# ---------- partición oficial (igual que worker_rev.py) ----------
po = pd.read_csv(W / "oficial" / "particion_oficial.csv")
po = po[po.split_oficial != ""]
part = {s: [(r.path, int(r.label)) for r in po[po.split_oficial == s].sort_values("path").itertuples()]
        for s in ("Training", "PublicTest", "PrivateTest")}
of_gen = g["build_balanced_train"](part["Training"], CFG["seed"], tag="_oficial")
rng = np.random.RandomState(CFG["seed"])
of_clean = [part["Training"][i] for i in sorted(rng.choice(len(part["Training"]), CFG["clean_train_eval_n"], replace=False))]

# ---------- RAF-DB en color: copias aumentadas guardadas en color ----------
def build_balanced_color(train_orig, seed, tag="_rafdb_color"):
    out_root = WORK / f"train_balanced{tag}_seed{seed}_size{CFG['aug_out_size']}"
    manifest = out_root / "manifest.csv"
    if manifest.exists():
        return [(r.path, int(r.label)) for r in pd.read_csv(manifest).itertuples()]
    random.seed(seed); torch.manual_seed(seed)
    by_class = {k: [p for p, l in train_orig if l == k] for k in range(g["NUM_CLASSES"])}
    target = max(len(v) for v in by_class.values()); rows = []
    for k, paths in by_class.items():
        d = out_root / g["CLASSES"][k]; d.mkdir(parents=True, exist_ok=True)
        for i in range(target - len(paths)):
            src = random.choice(paths)
            im = Image.open(src).convert("RGB")
            aug = g["augment_transform"](CFG["aug_out_size"] or im.size[0])(im)          # sin .convert("L")
            dst = d / f"{CFG['GENERATED_PREFIX']}{i}_{Path(src).stem}.jpg"
            aug.save(dst, quality=95); rows.append((str(dst), k, src))
    pd.DataFrame(rows, columns=["path", "label", "source"]).to_csv(manifest, index=False)
    return [(p, l) for p, l, _ in rows]

raf_gen_color = build_balanced_color(g["raf_train_orig"], CFG["seed"])
S_, MEAN, STD = CFG["img_size"], g["MEAN"], g["STD"]
COLOR_T = dict(
    train_transform=transforms.Compose([
        transforms.Resize((int(S_ * 256 / 224), int(S_ * 256 / 224))), transforms.RandomResizedCrop(S_, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(12), transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD), transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))]),
    eval_transform=transforms.Compose([transforms.Resize((S_, S_)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    tta_crop_transform=transforms.Compose([transforms.Resize((int(S_ * 236 / 224), int(S_ * 236 / 224))), transforms.CenterCrop(S_),
                                           transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))

# ---------- ConvNeXt-T con la misma interfaz que ResNet50CBAM (grupos conv1, layer1-4, cbam, head) ----------
class ConvNeXtCBAM(nn.Module):
    def __init__(self, num_classes=7, cbam_positions=("layer4",), pretrained=True):
        super().__init__()
        f = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None).features
        self.conv1, self.bn1 = f[0], nn.Identity()                       # stem (patchify 4x4) y sin BN aparte
        self.layer1, self.layer2, self.layer3, self.layer4 = f[1], nn.Sequential(f[2], f[3]), nn.Sequential(f[4], f[5]), nn.Sequential(f[6], f[7])
        mk = lambda c: g["CBAM"](c, CFG["cbam_reduction"], CFG["cbam_kernel"])
        self.cbam_layer2 = mk(192) if "layer2" in cbam_positions else None
        self.cbam_layer3 = mk(384) if "layer3" in cbam_positions else None
        self.cbam = mk(768) if "layer4" in cbam_positions else None
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(768), nn.Dropout(CFG["dropout1"]), nn.Linear(768, 512), nn.ReLU(inplace=True),
            nn.LayerNorm(512), nn.Dropout(CFG["dropout2"]), nn.Linear(512, num_classes))
    def cbam_modules(self):
        return [m for m in (self.cbam_layer2, self.cbam_layer3, self.cbam) if m is not None]
    def gradcam_layer(self):
        return self.cbam if self.cbam is not None else self.layer4
    def forward(self, x):
        x = self.layer2(self.layer1(self.conv1(x)))
        if self.cbam_layer2 is not None: x = self.cbam_layer2(x)
        x = self.layer3(x)
        if self.cbam_layer3 is not None: x = self.cbam_layer3(x)
        x = self.layer4(x)
        if self.cbam is not None: x = self.cbam(x)
        return self.classifier(self.pool(x))

CTX = {
    "ft_ablacion": dict(RUNS_DIR=W / "runs_ft_ablacion"),
    "sin_balanceo": dict(train_samples=list(g["train_orig"]), RUNS_DIR=W / "runs_sin_balanceo"),
    "oficial_pi1": dict(train_samples=part["Training"] + of_gen, val_samples=part["PublicTest"], test_samples=part["PrivateTest"],
                        train_clean_samples=of_clean, RUNS_DIR=W / "runs_oficial"),
    "rafdb_color": dict(train_samples=g["raf_train_orig"] + raf_gen_color, val_samples=g["raf_val"], test_samples=g["raf_test"],
                        train_clean_samples=g["raf_clean"], RUNS_DIR=W / "runs_rafdb_color", **COLOR_T),
    "convnext": dict(ResNet50CBAM=ConvNeXtCBAM, RUNS_DIR=W / "runs_convnext"),
}
for c in CTX.values():
    c["RUNS_DIR"].mkdir(exist_ok=True)

if prep:
    m = ConvNeXtCBAM(7, ("layer4",), True).to(g["device"]).eval()      # descarga los pesos y comprueba la forma
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 224, 224, device=g["device"]))
    assert out.shape == (2, 7)
    json.dump(dict(oficial={s: len(v) for s, v in part.items()}, raf_color_generadas=len(raf_gen_color),
                   sin_balanceo_train=len(g["train_orig"]), convnext_params=sum(p.numel() for p in m.parameters())),
              open(W / "cola_rev2" / "preparacion.json", "w"), indent=2)
    log(f"preparación lista: RAF-DB color {len(raf_gen_color)} generadas; ConvNeXt-T OK")
    sys.exit(0)

# ---------- variantes y configuración ----------
for v in ("sin_cbam", "sin_focal", "sin_mixup", "base"):
    VARIANTS[f"{v}_ft"] = dict(VARIANTS[v])
VARIANTS["completo_cnx"], VARIANTS["base_cnx"] = dict(VARIANTS["completo"]), dict(VARIANTS["base"])
STAGES_FT = [dict(start_epoch=0, train=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "cbam", "head"],
                  lr={"conv1": 1e-5, "bn1": 1e-5, "layer1": 1e-5, "layer2": 1e-5, "layer3": 1e-5, "layer4": 1e-5,
                      "cbam": 1e-4, "head": 3e-4})]

def configure(grp):
    CFG.clear(); CFG.update(copy.deepcopy(CFG0))
    if grp == "ft_ablacion":
        CFG["stages"] = copy.deepcopy(STAGES_FT)

def take(runs, v, s):
    d = W / runs / f"{v}_s{s}"
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / ".lock", "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close(); return None
    if (d / "meta.json").exists():
        f.close(); return None
    return f

failed = set()
while True:
    job = lock = None
    for jb in jobs():
        if jb not in failed and (lock := take(jb[1], jb[2], jb[3])):
            job = jb; break
    if job is None:
        break
    grp, runs, v, s = job
    log(f"inicio {grp} {v} s{s}")
    t0 = time.time()
    try:
        configure(grp)
        with data_context(**CTX[grp]):
            m = g["train_run_reanudable"](v, s, verbose=True)
        log(f"fin {grp} {v} s{s}: {m['epochs_run']} épocas, mejor {m['best_epoch']}, F1 val {m['best_val_f1']:.4f}, "
            f"{(time.time() - t0) / 3600:.2f} h")
    except Exception:
        failed.add(job)
        log(f"ERROR {grp} {v} s{s}\n{traceback.format_exc()}")
        torch.cuda.empty_cache()
    finally:
        configure("")
        lock.close()
log("sin trabajos pendientes")
