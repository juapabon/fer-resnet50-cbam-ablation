"""Análisis de la cola rev (experimentos pedidos en la revisión del artículo Frontiers).

Lee preds_test.csv de cada run y escribe en resultados_frontiers/ los ficheros rev_*.csv y rev_resumen.json.
Métricas sobre test con TTA (como en el artículo). Subconjuntos: clean = test de la copia sin las 647 imágenes con duplicado
(dHash<=8 y pHash-DCT<=8), clean10 = sin las 887 de la regla <=10, clean_of = PrivateTest sin sus 666 duplicados
en Training+PublicTest (duplicados_rev.py).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

W = Path.home() / "fer_frontiers_work"
OUT = Path.home() / "TESIS" / "Articulos Cientifico " / "resultados_frontiers"
C = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]; K = len(C)
S5, S10 = [42, 43, 44, 45, 46], list(range(42, 52))
DUP = {"clean": set(pd.read_csv(OUT / "test_con_duplicado_regla.csv").path),
       "clean10": set(pd.read_csv(OUT / "test_con_duplicado_regla10.csv").path),
       "clean_of": set(pd.read_csv(OUT / "oficial_test_con_duplicado_regla.csv").path)}

def mcnemar_exact(a, b):
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return 1.0 if n01 + n10 == 0 else min(1.0, 2 * stats.binom.cdf(min(n01, n10), n01 + n10, 0.5))

def macro_f1(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K); tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr, rc = np.nan_to_num(tp / cm.sum(0)), np.nan_to_num(tp / cm.sum(1))
        return np.nan_to_num(2 * pr * rc / (pr + rc)).mean()

def load(runs, v, seeds):
    out = {}
    for s in seeds:
        d = W / runs / f"{v}_s{s}"
        if (d / "meta.json").exists():
            out[s] = pd.read_csv(d / "preds_test.csv").sort_values("path").reset_index(drop=True)
    return out

def merge(*dicts):
    o = {}
    for d in dicts:
        o.update(d)
    return o

def metr(P, sub=None):
    """acc y F1 por semilla, y predicción del ensemble (media de probabilidades), sobre el subconjunto sub."""
    acc, f1, prob, y = {}, {}, [], None
    for s, df in sorted(P.items()):
        m = ~df.path.isin(DUP[sub]).values if sub else np.ones(len(df), bool)
        y = df.label.values[m]; pt = df[[f"ptta_{c}" for c in C]].values[m]; a = pt.argmax(1)
        acc[s], f1[s] = (a == y).mean(), macro_f1(y, a); prob.append(pt)
    return acc, f1, (np.mean(prob, 0).argmax(1), y)

def resumen(grupo, nombre, P, sub=None):
    a, f, (e, y) = metr(P, sub)
    a, f = np.array(list(a.values())), np.array(list(f.values()))
    return dict(grupo=grupo, modelo=nombre, test=sub or "full", n_seeds=len(a), n_test=len(y),
                acc_mean=a.mean(), acc_sd=a.std(ddof=1), f1_mean=f.mean(), f1_sd=f.std(ddof=1),
                acc_ens=(e == y).mean(), f1_ens=macro_f1(y, e))

def contraste(grupo, nA, A, nB, B, sub=None, margen=0.01):
    """A frente a B con semillas emparejadas: t pareada, Wilcoxon, TOST ±margen (acc) y McNemar de ensembles."""
    comun = sorted(set(A) & set(B))
    A, B = {s: A[s] for s in comun}, {s: B[s] for s in comun}
    aa, fa, (ea, ya) = metr(A, sub); ab, fb, (eb, yb) = metr(B, sub)
    assert (ya == yb).all()
    da = np.array([aa[s] - ab[s] for s in comun]); dfl = np.array([fa[s] - fb[s] for s in comun])
    n = len(da); se = da.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    p_tost = max(stats.t.sf((da.mean() + margen) / se, n - 1), stats.t.cdf((da.mean() - margen) / se, n - 1))
    try:
        pw = stats.wilcoxon(da).pvalue
    except ValueError:
        pw = np.nan
    return dict(grupo=grupo, A=nA, B=nB, test=sub or "full", seeds=n, d_acc=da.mean(), ic95_lo=da.mean() - tc * se,
                ic95_hi=da.mean() + tc * se, t_p_acc=stats.ttest_rel(da, np.zeros(n)).pvalue,
                wilcoxon_p_acc=pw, tost_p_1pt=p_tost, d_f1=dfl.mean(), t_p_f1=stats.ttest_rel(dfl, np.zeros(n)).pvalue,
                d_acc_ens=(ea == ya).mean() - (eb == yb).mean(), mcnemar_ens_p=mcnemar_exact(ea == ya, eb == yb))

R, T = [], []
# originales del artículo (balanceo con copias guardadas, partición de la copia)
comp, base = load("runs", "completo", S5), load("runs", "base", S5)

# (6) ajuste de la línea base: 12 configuraciones con s42 por F1 de validación; la mejor con 5 semillas
cfgs = json.load(open(W / "runs_ajuste" / "configuraciones.json"))
m0 = json.load(open(W / "runs/base_s42/meta.json"))
filas = [dict(cfg="cfg00", lr_mult=1.0, weight_decay=np.nan, dropout1=np.nan, dropout2=np.nan,
              val_f1_s42=m0["best_val_f1"], epochs=m0["epochs_run"])]
for k, c in cfgs.items():
    m = json.load(open(W / f"runs_ajuste/base_{k}_s42/meta.json"))
    filas.append(dict(cfg=k, **c, val_f1_s42=m["best_val_f1"], epochs=m["epochs_run"]))
AJ = pd.DataFrame(filas).sort_values("val_f1_s42", ascending=False); AJ.to_csv(OUT / "rev_ajuste_configuraciones.csv", index=False)
mejor = AJ.iloc[0].cfg
tuned = load("runs_ajuste", f"base_{mejor}", S5)
for sub in (None, "clean"):
    R += [resumen("articulo", "completo", comp, sub), resumen("articulo", "base", base, sub),
          resumen("ajuste", f"base_{mejor}", tuned, sub)]
    T += [contraste("ajuste", f"base_{mejor}", tuned, "base", base, sub),
          contraste("ajuste", "completo", comp, f"base_{mejor}", tuned, sub)]

# sensibilidad: test limpio con la regla <=10 (887 imágenes) para los modelos originales
R += [resumen("articulo", "completo", comp, "clean10"), resumen("articulo", "base", base, "clean10")]
T += [contraste("articulo", "completo", comp, "base", base, sub) for sub in (None, "clean", "clean10")]

# (2) partición oficial (Training / PublicTest / PrivateTest); el test es PrivateTest
oc, ob = load("runs_oficial", "completo", S5), load("runs_oficial", "base", S5)
for sub in (None, "clean_of"):
    R += [resumen("oficial", "completo", oc, sub), resumen("oficial", "base", ob, sub)]
    T += [contraste("oficial", "completo", oc, "base", ob, sub)]

# (8) muestreo ponderado sin copias guardadas frente al balanceo del artículo
mc, mb = load("runs_muestreo", "completo", S5), load("runs_muestreo", "base", S5)
for sub in (None, "clean"):
    R += [resumen("muestreo", "completo", mc, sub), resumen("muestreo", "base", mb, sub)]
    T += [contraste("muestreo", "completo", mc, "base", mb, sub),
          contraste("muestreo", "completo_muestreo", mc, "completo_balanceo", comp, sub),
          contraste("muestreo", "base_muestreo", mb, "base_balanceo", base, sub)]

# (4) 10 semillas (42-51)
c10, b10 = merge(comp, load("runs_semillas", "completo", S10[5:])), merge(base, load("runs_semillas", "base", S10[5:]))
for sub in (None, "clean"):
    R += [resumen("semillas10", "completo", c10, sub), resumen("semillas10", "base", b10, sub)]
    T += [contraste("semillas10", "completo", c10, "base", b10, sub)]

# (7) ajuste fino completo desde la época 1 (sin descongelamiento progresivo)
ft = load("runs_ajuste_fino", "completo_ft", S5)
for sub in (None, "clean"):
    R += [resumen("ajuste_fino", "completo_ft", ft, sub)]
    T += [contraste("ajuste_fino", "completo_ft", ft, "completo", comp, sub),
          contraste("ajuste_fino", "completo_ft", ft, "base", base, sub),
          contraste("ajuste_fino", "completo_ft", ft, f"base_{mejor}", tuned, sub)]

RES, CON = pd.DataFrame(R), pd.DataFrame(T)
RES.to_csv(OUT / "rev_resultados.csv", index=False); CON.to_csv(OUT / "rev_contrastes.csv", index=False)

# épocas, tiempos y reanudaciones de cada run
tm = []
for runs in ("runs_ajuste", "runs_oficial", "runs_muestreo", "runs_semillas", "runs_ajuste_fino"):
    for f in sorted((W / runs).glob("*/meta.json")):
        m = json.load(open(f))
        tm.append(dict(runs=runs, run=f.parent.name, epochs=m["epochs_run"], best_epoch=m["best_epoch"],
                       val_f1=m["best_val_f1"], horas=m["train_time_s"] / 3600,
                       reanudada=bool(m.get("reanudada_tras_epocas"))))
TM = pd.DataFrame(tm); TM.to_csv(OUT / "rev_runs.csv", index=False)
json.dump(dict(mejor_cfg=mejor, mejor_cfg_param=cfgs.get(mejor), n_runs=len(TM), horas_gpu=float(TM.horas.sum()),
               tamanos_oficial=json.load(open(W / "oficial/tamanos.json"))), open(OUT / "rev_resumen.json", "w"), indent=2)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(AJ.round(4).to_string(index=False)); print()
print(RES.round(4).to_string(index=False)); print()
print(CON.round(4).to_string(index=False)); print()
print(TM.groupby("runs")[["epochs", "horas"]].agg(["mean", "min", "max"]).round(2)); print("reanudadas:", TM.reanudada.sum())
