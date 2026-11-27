"""Análisis de la cola rev2 (comentarios 1, 6, 7 y 9 del informe Q1 del artículo Frontiers).

Mismas métricas y contrastes que analisis_rev.py (TTA, t pareada, Wilcoxon, TOST ±1 punto, McNemar de ensembles),
sobre la prueba completa y la limpia. Escribe en resultados_frontiers/ rev2_resultados.csv, rev2_contrastes.csv,
rev2_runs.csv y rev2_resumen.json.
  - clean     = test de la copia sin las 647 imágenes con duplicado (regla dHash<=8 y pHash-DCT<=8)
  - clean_of  = PrivateTest oficial sin sus 666 duplicados en Training+PublicTest
  - clean_raf = test de RAF-DB sin las imágenes con duplicado en train/val (misma regla; 141 en el artículo)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

W = Path.home() / "fer_frontiers_work"
OUT = Path.home() / "TESIS" / "Articulos Cientifico " / "resultados_frontiers"
# funciones comunes de analisis_rev.py (mcnemar_exact, macro_f1, load, merge, metr, resumen, contraste, DUP, S5)
exec(open(W / "tmp/rev/analisis_rev.py", encoding="utf-8").read().split("R, T = [], []")[0])
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def hashes_rafdb(k):
    d = pd.read_csv(W / f"hashes_rafdb_{k}_d.csv", dtype={"phash": str})
    e = pd.read_csv(W / f"hashes_extra_rafdb_{k}.csv", dtype=str)
    assert list(d.path) == list(e.path)
    return pd.DataFrame(dict(path=d.path, dhash=[np.uint64(int(v)) for v in d.phash],
                             **{c: [np.uint64(int(v, 16)) for v in e[c]] for c in ("dhash_flip", "phash_dct", "phash_dct_flip")}))

def ham(q, r):
    x = np.bitwise_xor(q[:, None], r[None, :])
    return POP[x.view(np.uint8)].reshape(len(q), len(r), 8).sum(2, dtype=np.uint8)

def flagged(Q, R, td=8, tp=8, chunk=256):
    rd, rp = R.dhash.values.astype(np.uint64), R.phash_dct.values.astype(np.uint64)
    hit = np.zeros(len(Q), bool)
    for i in range(0, len(Q), chunk):
        q = Q.iloc[i:i + chunk]
        a = (ham(q.dhash.values.astype(np.uint64), rd) <= td) & (ham(q.phash_dct.values.astype(np.uint64), rp) <= tp)
        b = (ham(q.dhash_flip.values.astype(np.uint64), rd) <= td) & (ham(q.phash_dct_flip.values.astype(np.uint64), rp) <= tp)
        hit[i:i + chunk] = (a | b).any(1)
    return hit

def holm(p):
    p = np.asarray(p, float); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def con_holm(filas, campos=("t_p_acc", "mcnemar_ens_p")):
    for c in campos:
        for f, a in zip(filas, holm([f[c] for f in filas])):
            f[f"holm_{c}"] = a
    return filas

# duplicados de RAF-DB (test frente a train+val), misma regla que la sección 18 del cuaderno
QR = hashes_rafdb("test"); fr = flagged(QR, hashes_rafdb("trainval"))
pd.DataFrame(dict(path=QR.path[fr])).to_csv(OUT / "rafdb_test_con_duplicado_regla.csv", index=False)
DUP["clean_raf"] = set(QR.path[fr]); n_dup_raf = int(fr.sum())
print("RAF-DB test con duplicado:", n_dup_raf, "(artículo:", json.load(open(OUT / "rafdb_duplicados.json"))["test_with_dup_in_trainval"], ")")

R, T = [], []
comp, base = load("runs", "completo", S5), load("runs", "base", S5)

# (1) eliminaciones con ajuste fino completo desde la época 1 (sin descongelamiento progresivo)
ft = {"completo_ft": load("runs_ajuste_fino", "completo_ft", S5)}
for v in ("sin_cbam", "sin_focal", "sin_mixup", "base"):
    ft[f"{v}_ft"] = load("runs_ft_ablacion", f"{v}_ft", S5)
for sub in (None, "clean"):
    R += [resumen("ft_ablacion", k, P, sub) for k, P in ft.items()]
    T += con_holm([contraste("ft_ablacion", k, ft[k], "completo_ft", ft["completo_ft"], sub) for k in ft if k != "completo_ft"])
    T += [contraste("ft_vs_progresivo", "base_ft", ft["base_ft"], "base", base, sub)]
    for v in ("sin_cbam", "sin_focal", "sin_mixup"):
        T += [contraste("ft_vs_progresivo", f"{v}_ft", ft[f"{v}_ft"], v, load("runs", v, S5), sub)]

# (9) pérdida focal frente a entropía cruzada sin balanceo (pesos inversos a la frecuencia sobre train_orig)
nb = {v: load("runs_sin_balanceo", v, S5) for v in ("completo", "sin_focal")}
bal_sf = load("runs", "sin_focal", S5)
for sub in (None, "clean"):
    R += [resumen("sin_balanceo", k, P, sub) for k, P in nb.items()]
    T += [contraste("sin_balanceo", "completo", nb["completo"], "sin_focal", nb["sin_focal"], sub),
          contraste("balanceo", "completo", comp, "sin_focal", bal_sf, sub),
          contraste("sinbal_vs_bal", "completo_sinbal", nb["completo"], "completo_bal", comp, sub),
          contraste("sinbal_vs_bal", "sin_focal_sinbal", nb["sin_focal"], "sin_focal_bal", bal_sf, sub)]

# (7) ablación PI1 en la partición oficial: cada variante frente a completo, con Holm
VO = ("base", "sin_cbam", "cbam_layer2", "cbam_layer3", "cbam_layer3_4", "cbam_layer2_3_4")
of = {v: load("runs_oficial", v, S5) for v in ("completo",) + VO}
for sub in (None, "clean_of"):
    R += [resumen("oficial_pi1", k, P, sub) for k, P in of.items()]
    T += con_holm([contraste("oficial_pi1", v, of[v], "completo", of["completo"], sub) for v in VO])

# (6) RAF-DB en color frente a gris, y ConvNeXt-T
rg = {v: load("runs_rafdb", v, S5) for v in ("completo", "base")}
rc = {v: load("runs_rafdb_color", v, S5) for v in ("completo", "base")}
for sub in (None, "clean_raf"):
    R += [resumen("rafdb_gris", k, P, sub) for k, P in rg.items()] + [resumen("rafdb_color", k, P, sub) for k, P in rc.items()]
    T += [contraste("rafdb_color", "completo", rc["completo"], "base", rc["base"], sub),
          contraste("rafdb_gris", "completo", rg["completo"], "base", rg["base"], sub),
          contraste("rafdb_color_vs_gris", "completo_color", rc["completo"], "completo_gris", rg["completo"], sub),
          contraste("rafdb_color_vs_gris", "base_color", rc["base"], "base_gris", rg["base"], sub)]
cn = {v: load("runs_convnext", f"{v}_cnx", S5) for v in ("completo", "base")}
for sub in (None, "clean"):
    R += [resumen("convnext", f"{k}_cnx", P, sub) for k, P in cn.items()]
    T += [contraste("convnext", "completo_cnx", cn["completo"], "base_cnx", cn["base"], sub),
          contraste("convnext", "completo_cnx", cn["completo"], "completo_resnet", comp, sub),
          contraste("convnext", "base_cnx", cn["base"], "base_resnet", base, sub)]

RES, CON = pd.DataFrame(R), pd.DataFrame(T)
RES.to_csv(OUT / "rev2_resultados.csv", index=False); CON.to_csv(OUT / "rev2_contrastes.csv", index=False)

# épocas, tiempos, parámetros y reanudaciones de las runs de rev2 (en runs_oficial, completo y base son de la cola rev)
tm = []
for runs in ("runs_ft_ablacion", "runs_sin_balanceo", "runs_oficial", "runs_rafdb_color", "runs_convnext"):
    for fm in sorted((W / runs).glob("*/meta.json")):
        if runs == "runs_oficial" and fm.parent.name.rsplit("_s", 1)[0] in ("completo", "base"):
            continue
        m = json.load(open(fm))
        tm.append(dict(runs=runs, run=fm.parent.name, epochs=m["epochs_run"], best_epoch=m["best_epoch"], val_f1=m["best_val_f1"],
                       horas=m["train_time_s"] / 3600, params=m.get("trainable_params_final"), sec_epoca=m.get("sec_per_epoch"),
                       gpu_gb=m.get("peak_gpu_mem_gb"), reanudada=bool(m.get("reanudada_tras_epocas"))))
TM = pd.DataFrame(tm); TM.to_csv(OUT / "rev2_runs.csv", index=False)
json.dump(dict(n_runs=len(TM), horas_gpu=float(TM.horas.sum()), rafdb_test_duplicados=n_dup_raf,
               rafdb_test_limpio=int(len(QR) - n_dup_raf), reanudadas=int(TM.reanudada.sum())),
          open(OUT / "rev2_resumen.json", "w"), indent=2)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(RES.round(4).to_string(index=False)); print()
cols = ["grupo", "A", "B", "test", "seeds", "d_acc", "ic95_lo", "ic95_hi", "t_p_acc", "wilcoxon_p_acc", "tost_p_1pt", "d_f1", "t_p_f1",
        "d_acc_ens", "mcnemar_ens_p", "holm_t_p_acc", "holm_mcnemar_ens_p"]
print(CON[cols].round(4).to_string(index=False)); print()
print(TM.groupby("runs")[["epochs", "best_epoch", "horas", "sec_epoca", "gpu_gb"]].mean().round(2))
print(TM[TM.run.str.endswith("s42")][["run", "params"]].to_string(index=False))
