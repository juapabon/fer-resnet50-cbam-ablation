"""Worker de los experimentos adicionales (sin repetidas / RAF-DB) del artículo Frontiers.

Ejecuta las celdas del cuaderno hasta train_run (celda 22) y las celdas marcadas con '#@setup:<grupo>',
y luego reclama trabajos (variante, semilla) con mkdir atómico en <cola>/claims.
Uso: python worker_ext.py <grupo> <id> [--prep]      grupo = dedup | rafdb
"""
import json, os, sys, time, traceback
from pathlib import Path

NB = "FER_ResNet50_CBAM_Frontiers.ipynb"
grp, wid = sys.argv[1], sys.argv[2]
prep = "--prep" in sys.argv
BASE = Path.home() / "fer_frontiers_work" / f"cola_{grp}"
CLAIMS = BASE / "claims"
NUM_WORKERS = int(os.environ.get("FER_NUM_WORKERS", "5"))

def log(msg):
    print(f"{time.strftime('%F %T')} [{grp} w{wid}] {msg}", flush=True)

os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(Path(NB).parent)
cells = json.load(open(NB))["cells"]
g = {"__name__": "__main__"}
from IPython.display import display
g["display"] = display
needed = {"dedup": ["dup", "dedup"], "rafdb": ["rafdb"]}[grp]
for i, c in enumerate(cells):
    src = "".join(c["source"])
    tag = src.split("\n", 1)[0].replace("#@setup:", "") if src.startswith("#@setup:") else None
    if c["cell_type"] != "code" or not (i <= 22 or tag in needed):
        continue
    exec(compile(src, f"<celda {i}>", "exec"), g)
    if i == 4:
        g["CFG"]["num_workers"] = NUM_WORKERS
log("setup listo")
if prep:
    sys.exit(0)

ctx = g["DEDUP_CTX"] if grp == "dedup" else g["RAF_CTX"]
variants = g["DEDUP_VARIANTS"] if grp == "dedup" else g["RAF_VARIANTS"]
seeds = g["DEDUP_SEEDS"] if grp == "dedup" else g["RAF_SEEDS"]
runs = ctx["RUNS_DIR"]
CLAIMS.mkdir(parents=True, exist_ok=True)
for s in seeds:
    for v in variants:
        if (runs / f"{v}_s{s}" / "meta.json").exists():
            continue
        try:
            os.mkdir(CLAIMS / f"{v}_s{s}")
        except FileExistsError:
            continue
        log(f"inicio {v} s{s}")
        t0 = time.time()
        try:
            with g["data_context"](**ctx):
                m = g["train_run"](v, s, verbose=True)
            log(f"fin {v} s{s}: {m['epochs_run']} épocas, mejor {m['best_epoch']}, F1 val {m['best_val_f1']:.4f}, "
                f"{(time.time() - t0) / 3600:.2f} h")
        except Exception:
            log(f"ERROR {v} s{s}\n{traceback.format_exc()}")
            import torch; torch.cuda.empty_cache()
log("sin trabajos pendientes")
