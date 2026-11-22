"""Informe Q1 (comentarios 2 y 3): composición por clases de la prueba limpia y IC del acuerdo entre anotadores.

1) Para los modelos originales (completo y base, semillas 42-46, TTA), descompone la caída de exactitud entre la prueba
   completa y la limpia: full = (1-f)·clean + f·dup, así que full - clean = f·(dup - clean). La parte de composición es
   f·(dup_esperada - clean), donde dup_esperada es la exactitud que tendrían las imágenes duplicadas con la exactitud por
   clase de las limpias; el resto, f·(dup - dup_esperada), es la ventaja dentro de cada clase. También da la exactitud
   de la prueba limpia reponderada a la distribución de clases de la prueba completa.
2) Acuerdo entre anotadores: IC bootstrap del kappa de 3 categorías e IC de Clopper-Pearson del acuerdo binario.
Salida: resultados_frontiers/rev_composicion_clases.csv y rev_composicion.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

W = Path.home() / "fer_frontiers_work"
OUT = Path.home() / "TESIS" / "Articulos Cientifico " / "resultados_frontiers"
C = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DUP = set(pd.read_csv(OUT / "test_con_duplicado_regla.csv").path)

rows, dec = [], []
for v in ("completo", "base"):
    for s in range(42, 47):
        df = pd.read_csv(W / "runs" / f"{v}_s{s}" / "preds_test.csv")
        y = df.label.values; ok = df[[f"ptta_{c}" for c in C]].values.argmax(1) == y; d = df.path.isin(DUP).values
        f = d.mean(); acc_full, acc_clean, acc_dup = ok.mean(), ok[~d].mean(), ok[d].mean()
        rc = np.array([ok[(~d) & (y == k)].mean() for k in range(7)])          # exactitud por clase, limpias
        rd = np.array([ok[d & (y == k)].mean() if (d & (y == k)).any() else np.nan for k in range(7)])
        wf = np.bincount(y, minlength=7) / len(y); wd = np.bincount(y[d], minlength=7) / d.sum()
        dup_exp = (wd * rc).sum()
        dec.append(dict(modelo=v, semilla=s, f_dup=f, acc_full=acc_full, acc_clean=acc_clean, acc_dup=acc_dup,
                        acc_clean_reponderada=(wf * rc).sum(), acc_dup_esperada=dup_exp,
                        caida=acc_full - acc_clean, caida_composicion=f * (dup_exp - acc_clean), caida_intra_clase=f * (acc_dup - dup_exp)))
        for k in range(7):
            rows.append(dict(modelo=v, semilla=s, clase=C[k], n_test=int((y == k).sum()), n_dup=int((d & (y == k)).sum()),
                             acc_dup=rd[k], acc_clean=rc[k]))
DEC = pd.DataFrame(dec); CL = pd.DataFrame(rows)
cls = CL.groupby(["modelo", "clase"]).agg(n_test=("n_test", "first"), n_dup=("n_dup", "first"),
                                          acc_dup=("acc_dup", "mean"), acc_clean=("acc_clean", "mean")).reset_index()
cls["pct_dup"] = 100 * cls.n_dup / cls.n_test
cls.to_csv(OUT / "rev_composicion_clases.csv", index=False)
res = {}
for v, g in DEC.groupby("modelo"):
    m = g.mean(numeric_only=True); sd = g.std(numeric_only=True)
    res[v] = {k: [round(100 * m[k], 2), round(100 * sd[k], 2)] for k in
              ("acc_full", "acc_clean", "acc_dup", "acc_clean_reponderada", "acc_dup_esperada", "caida", "caida_composicion", "caida_intra_clase")}
    # ventaja dentro de clase por semilla: t de una muestra
    res[v]["p_intra_clase"] = float(stats.ttest_1samp(g.caida_intra_clase, 0).pvalue)

# acuerdo entre anotadores
R = pd.read_csv(OUT / "revision_duplicados_muestra.csv", keep_default_na=False).set_index("id")
k = pd.read_csv(OUT / "revision_ciega_120_clave.csv").merge(pd.read_csv(OUT / "revision_ciega_120_anotador2.csv", keep_default_na=False), on="hoja_id")
r1, r2 = R.loc[k.par_id, "verdict"].values, k.verdict.values
rng = np.random.RandomState(0); bs = []
for _ in range(5000):
    i = rng.randint(0, len(k), len(k))
    if len(set(r1[i]) | set(r2[i])) > 1:
        bs.append(cohen_kappa_score(r1[i], r2[i], labels=["same", "different", "uncertain"]))
agree = int(((r1 == "same") == (r2 == "same")).sum()); n = len(k)
res["kappa"] = dict(n=n, kappa_3=float(cohen_kappa_score(r1, r2, labels=["same", "different", "uncertain"])),
                    kappa_3_ic95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    acuerdo_binario=f"{agree}/{n}", acuerdo_binario_ic95_cp=[float(stats.beta.ppf(0.025, agree, n - agree + 1)) if agree < n or True else None, 1.0])
json.dump(res, open(OUT / "rev_composicion.json", "w"), indent=2)
print(json.dumps(res, indent=1))
print(cls[cls.modelo == "completo"].round(3).to_string(index=False))
