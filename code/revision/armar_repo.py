"""Arma el repositorio público del artículo en ~/fer_frontiers_repo (sin imágenes de los conjuntos de datos ni pesos).

- code/: cuadernos (sin ejecutar y ejecutado) y scripts de las colas (ext y revisión)
- results/: CSV/JSON/tablas LaTeX y figuras del artículo; se excluyen las láminas de revisión y la hoja ciega,
  que contienen imágenes de FER2013
- runs/<grupo>/<run>/: meta.json, history.json y preds_test.csv de cada entrenamiento
- splits/: partición oficial reconstruida (con el índice de FERPlus) y lista de duplicados
Las rutas absolutas del servidor se sustituyen por rutas relativas al conjunto de datos (FER2013/..., RAF-DB/...).
Reejecutable: sólo copia y reescribe; los pesos (.pth) van a Zenodo.
"""
import json, re, shutil
from pathlib import Path

H = Path.home(); SRC = H / "TESIS/Articulos Cientifico "; W = H / "fer_frontiers_work"; R = H / "fer_frontiers_repo"
SUBS = [(str(H / "TESIS/Caras/Nuevo_dataset") + "/", "FER2013/"),
        (str(H / "TESIS/datasets_emociones/RAF-DB/DATASET") + "/", "RAF-DB/"),
        (str(W / "train_balanced") , "WORK_DIR/train_balanced"),
        (str(W) + "/", "WORK_DIR/"), (str(SRC) + "/", ""), (str(H) + "/", "~/")]
TEXT = {".csv", ".json", ".tex", ".md", ".txt", ".py", ".sh", ".ipynb", ".bib", ".cff"}

def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT:
        t = src.read_text(encoding="utf8", errors="replace")
        for a, b in SUBS:
            t = t.replace(a, b)
        dst.write_text(t, encoding="utf8")
    else:
        shutil.copy2(src, dst)

for f in ("FER_ResNet50_CBAM_Frontiers.ipynb", "FER_ResNet50_CBAM_Frontiers_completo_ejecutado.ipynb"):
    put(SRC / f, R / "code" / f)
for f in ("worker_ext.py", "lanzar_ext.sh", "orquestador_ext.sh"):
    put(W / "tmp" / f, R / "code" / "queues" / f)
for f in sorted((W / "tmp" / "rev").glob("*")):
    if f.suffix in (".py", ".sh"):
        put(f, R / "code" / "revision" / f.name)
skip_dirs = {"revision_duplicados"}
for f in sorted((SRC / "resultados_frontiers").rglob("*")):
    rel = f.relative_to(SRC / "resultados_frontiers")
    if f.is_dir() or skip_dirs & set(rel.parts) or f.name.startswith("revision_ciega_120.html"):
        continue
    put(f, R / "results" / rel)
n = 0
for d in sorted(W.glob("runs*")):
    for run in sorted(p for p in d.iterdir() if (p / "meta.json").exists()):
        for f in ("meta.json", "history.json", "preds_test.csv", "configuraciones.json"):
            if (run / f).exists():
                put(run / f, R / "runs" / d.name / run.name / f)
        n += 1
    if (d / "configuraciones.json").exists():
        put(d / "configuraciones.json", R / "runs" / d.name / "configuraciones.json")
for f in ("particion_oficial.csv", "resumen_emparejamiento.json", "tamanos.json"):
    if (W / "oficial" / f).exists():
        put(W / "oficial" / f, R / "splits" / f)
print("runs copiados:", n)
print("tamaño:", sum(f.stat().st_size for f in R.rglob("*") if f.is_file() and ".git" not in f.parts) / 1e6, "MB")
bad = [str(f) for f in R.rglob("*") if f.is_file() and f.suffix in TEXT and ".git" not in f.parts and "/home/jpabon" in f.read_text(errors="ignore")]
print("ficheros con rutas absolutas restantes:", bad[:5])
