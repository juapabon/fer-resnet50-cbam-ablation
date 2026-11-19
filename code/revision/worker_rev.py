"""Worker de los experimentos pedidos en la revisión del artículo Frontiers (cola única con prioridades).

Grupos, en orden de prioridad (todos con el protocolo y los hiperparámetros del artículo salvo lo indicado):
  ajuste      línea base con 11 configuraciones aleatorias (semilla 42) + la por defecto (runs/base_s42);
              después, la mejor por F1 macro de validación con las semillas 43-46            -> runs_ajuste
  oficial     completo y base x 5 semillas en la partición oficial (Training / PublicTest / PrivateTest),
              reconstruida emparejando con FERPlus (oficial/particion_oficial.csv)        -> runs_oficial
  muestreo    completo y base x 5 semillas con muestreo ponderado por clase sobre las originales,
              sin copias aumentadas guardadas; mismas iteraciones por época que el balanceo -> runs_muestreo
  semillas    completo y base con las semillas 47-51 (para tener 10 semillas)             -> runs_semillas
  ajuste_fino completo x 5 semillas con toda la red entrenable desde la época 1            -> runs_ajuste_fino

Uso: python worker_rev.py <id> [--prep | --pending]
"""
import copy, json, os, sys, time, traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np

NB = "FER_ResNet50_CBAM_Frontiers.ipynb"
W = Path.home() / "fer_frontiers_work"
BASE = W / "cola_rev"; CLAIMS = BASE / "claims"
wid = sys.argv[1]
prep, pending_only = "--prep" in sys.argv, "--pending" in sys.argv
NUM_WORKERS = int(os.environ.get("FER_NUM_WORKERS", "5"))
SEEDS, EXTRA_SEEDS = [42, 43, 44, 45, 46], [47, 48, 49, 50, 51]
N_TUNE, TUNE_SEED = 11, 2026

def log(msg):
    print(f"{time.strftime('%F %T')} [rev w{wid}] {msg}", flush=True)

def tune_configs():
    """11 configuraciones aleatorias para la línea base (la 0 es la del artículo). Fijas: semilla TUNE_SEED."""
    rng = np.random.RandomState(TUNE_SEED); out = {}
    for k in range(1, N_TUNE + 1):
        out[f"cfg{k:02d}"] = dict(lr_mult=float(np.exp(rng.uniform(np.log(1 / 3), np.log(3)))),
                                  weight_decay=float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-2)))),
                                  dropout1=float(rng.choice([0.3, 0.4, 0.5, 0.6])),
                                  dropout2=float(rng.choice([0.2, 0.3, 0.4, 0.5])))
    return out

TUNE = tune_configs()

def done(runs, name, seed):
    return (W / runs / f"{name}_s{seed}" / "meta.json").exists()

def best_tune():
    """Mejor configuración de la línea base por F1 macro de validación (semilla 42); None si falta alguna."""
    scores = {"cfg00": json.load(open(W / "runs" / "base_s42" / "meta.json"))["best_val_f1"]}
    for k in TUNE:
        f = W / "runs_ajuste" / f"base_{k}_s42" / "meta.json"
        if not f.exists():
            return None
        scores[k] = json.load(open(f))["best_val_f1"]
    return max(scores, key=scores.get), scores

def jobs():
    """Lista (grupo, variante, semilla) pendiente en orden de prioridad, y si hay trabajos bloqueados."""
    J, blocked = [], False
    J += [("ajuste", f"base_{k}", 42) for k in TUNE if not done("runs_ajuste", f"base_{k}", 42)]
    b = best_tune()
    if b is None:
        blocked = True
    elif b[0] != "cfg00":
        J += [("ajuste", f"base_{b[0]}", s) for s in SEEDS[1:] if not done("runs_ajuste", f"base_{b[0]}", s)]
    for grp, runs, variants, seeds in (("oficial", "runs_oficial", ["completo", "base"], SEEDS),
                                       ("muestreo", "runs_muestreo", ["completo", "base"], SEEDS),
                                       ("semillas", "runs_semillas", ["completo", "base"], EXTRA_SEEDS),
                                       ("ajuste_fino", "runs_ajuste_fino", ["completo_ft"], SEEDS)):
        J += [(grp, v, s) for s in seeds for v in variants if not done(runs, v, s)]
    return J, blocked

if pending_only:
    J, blocked = jobs()
    print(len(J) + (1 if blocked else 0))
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
import torch, pandas as pd
CFG, VARIANTS = g["CFG"], g["VARIANTS"]
CFG0 = copy.deepcopy(CFG)
WORK = g["WORK"]

@contextmanager
def data_context(**kw):
    old = {k: g[k] for k in kw}
    g.update(kw)
    try:
        yield
    finally:
        g.update(old)

# ---------- partición oficial ----------
po = pd.read_csv(W / "oficial" / "particion_oficial.csv")
po = po[po.split_oficial != ""]
part = {s: [(r.path, int(r.label)) for r in po[po.split_oficial == s].sort_values("path").itertuples()]
        for s in ("Training", "PublicTest", "PrivateTest")}
of_gen = g["build_balanced_train"](part["Training"], CFG["seed"], tag="_oficial")
rng = np.random.RandomState(CFG["seed"])
of_clean = [part["Training"][i] for i in sorted(rng.choice(len(part["Training"]), CFG["clean_train_eval_n"], replace=False))]
CTX = {
    "oficial": dict(train_samples=part["Training"] + of_gen, val_samples=part["PublicTest"], test_samples=part["PrivateTest"],
                    train_clean_samples=of_clean, RUNS_DIR=W / "runs_oficial"),
    "muestreo": dict(train_samples=list(g["train_orig"]), RUNS_DIR=W / "runs_muestreo"),
    "semillas": dict(RUNS_DIR=W / "runs_semillas"),
    "ajuste": dict(RUNS_DIR=W / "runs_ajuste"),
    "ajuste_fino": dict(RUNS_DIR=W / "runs_ajuste_fino"),
}
for c in CTX.values():
    c["RUNS_DIR"].mkdir(exist_ok=True)
json.dump(TUNE, open(W / "runs_ajuste" / "configuraciones.json", "w"), indent=2)
json.dump({**{s: len(v) for s, v in part.items()}, "Training_generadas": len(of_gen)},
          open(W / "oficial" / "tamanos.json", "w"), indent=2)
N_BALANCED = len(g["train_samples"])   # iteraciones por época del balanceo original (originales + generadas)
log(f"setup listo: oficial {[(s, len(v)) for s, v in part.items()]}, generadas {len(of_gen)}; muestreo {N_BALANCED} por época")
if prep:
    sys.exit(0)

# ---------- variantes y parches ----------
VARIANTS["completo_ft"] = dict(VARIANTS["completo"])
for k in TUNE:
    VARIANTS[f"base_{k}"] = dict(VARIANTS["base"])
STAGES_FT = [dict(start_epoch=0, train=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "cbam", "head"],
                  lr={"conv1": 1e-5, "bn1": 1e-5, "layer1": 1e-5, "layer2": 1e-5, "layer3": 1e-5, "layer4": 1e-5,
                      "cbam": 1e-4, "head": 3e-4})]
orig_make_loader, orig_make_criterion = g["make_loader"], g["make_criterion"]
SAMPLER = {"on": False}

def make_loader_w(samples, transform, shuffle, seed=0, tta=False, drop_last=False):
    """Con SAMPLER activo, el cargador de entrenamiento muestrea con reposición y peso 1/n_clase."""
    if not (SAMPLER["on"] and shuffle):
        return orig_make_loader(samples, transform, shuffle, seed, tta, drop_last)
    y = np.array([l for _, l in samples]); w = 1.0 / np.bincount(y, minlength=g["NUM_CLASSES"])[y]
    gen = torch.Generator(); gen.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), N_BALANCED, replacement=True, generator=gen)
    return torch.utils.data.DataLoader(g["ListDataset"](samples, transform, tta), batch_size=CFG["batch_size"], sampler=sampler,
                                       num_workers=CFG["num_workers"], pin_memory=True, drop_last=drop_last,
                                       worker_init_fn=g["seed_worker"], generator=gen)

def make_criterion_w(kind):
    """Con muestreo ponderado las clases ya llegan equilibradas: pérdida sin pesos (como con el balanceo)."""
    if not SAMPLER["on"]:
        return orig_make_criterion(kind)
    return g["FocalLoss"](None, CFG["focal_gamma"]) if kind == "focal" else torch.nn.CrossEntropyLoss()

g["make_loader"], g["make_criterion"] = make_loader_w, make_criterion_w

def configure(grp, variant):
    CFG.clear(); CFG.update(copy.deepcopy(CFG0))
    SAMPLER["on"] = grp == "muestreo"
    if variant.startswith("base_cfg"):
        t = TUNE[variant[5:]]
        CFG["stages"] = [dict(s, lr={k: v * t["lr_mult"] for k, v in s["lr"].items()}) for s in CFG["stages"]]
        CFG.update(weight_decay=t["weight_decay"], dropout1=t["dropout1"], dropout2=t["dropout2"])
    if variant == "completo_ft":
        CFG["stages"] = copy.deepcopy(STAGES_FT)

# ---------- bucle de trabajos ----------
CLAIMS.mkdir(parents=True, exist_ok=True)
while True:
    J, blocked = jobs()
    job = None
    for grp, v, s in J:
        try:
            os.mkdir(CLAIMS / f"{grp}__{v}_s{s}"); job = (grp, v, s); break
        except FileExistsError:
            continue
    if job is None:
        if blocked:
            time.sleep(300); continue
        break
    grp, v, s = job
    log(f"inicio {grp} {v} s{s}")
    t0 = time.time()
    try:
        configure(grp, v)
        with data_context(**CTX[grp]):
            m = g["train_run"](v, s, verbose=True)
        log(f"fin {grp} {v} s{s}: {m['epochs_run']} épocas, mejor {m['best_epoch']}, F1 val {m['best_val_f1']:.4f}, "
            f"{(time.time() - t0) / 3600:.2f} h")
    except Exception:
        log(f"ERROR {grp} {v} s{s}\n{traceback.format_exc()}")
        torch.cuda.empty_cache()
    finally:
        configure("", "")
log("sin trabajos pendientes")
