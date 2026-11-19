"""Punto 3 de la revisión: elección de la regla de duplicados con una mitad de los pares revisados y evaluación con la otra.

Para cada partición aleatoria 50/50 (estratificada por muestra y banda) se elige, en la mitad de ajuste, la regla
'dHash <= a y pHash-DCT <= b' de la rejilla que maximiza el F1 ponderado (estimador estratificado: cada par pesa
tamaño de su banda / pares revisados); después se estiman su precisión y su sensibilidad en la mitad de evaluación.
Se repite 2000 veces. 'uncertain' cuenta como no duplicado, como en el artículo.
Salidas en resultados_frontiers/: revision_mitades_resumen.json y revision_mitades_reglas.csv
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path.home() / "TESIS/Articulos Cientifico /resultados_frontiers"
R = pd.read_csv(OUT / "revision_duplicados_muestra.csv", keep_default_na=False)
B = pd.read_csv(OUT / "revision_duplicados_por_banda.csv", dtype={"band": str})
G = pd.read_csv(OUT / "duplicados_rejilla_reglas.csv", dtype=str)
w_band = {(r.hash, r.band): r.test_images_in_band / r.reviewed for r in B.itertuples()}
R["w"] = [w_band[(s, str(b))] for s, b in zip(R["sample"], R.band)]
R["y"] = (R.verdict == "same").astype(int)
TH = [0, 2, 4, 6, 8, 10, 64]
RULES = [(a, b) for a in TH for b in TH if not (a == 64 and b == 64)]
flagged = {(int(r.dhash_max.replace("-", "64")), int(r.phash_max.replace("-", "64"))): int(r.test_flagged) for r in G.itertuples()}
d, p, y, w = R.dhash_d.values, R.phash_d.values, R.y.values, R.w.values
F = {r: ((d <= r[0]) & (p <= r[1])).astype(int) for r in RULES}

def prf(mask, rule):
    f, yy, ww = F[rule][mask], y[mask], w[mask]
    tp = (ww * f * yy).sum(); pf = (ww * f).sum(); py = (ww * yy).sum()
    prec = tp / pf if pf else np.nan; rec = tp / py if py else np.nan
    return prec, rec, (2 * prec * rec / (prec + rec) if prec and rec and prec + rec else 0.0)

strata = R.groupby(["sample", "band"]).indices
rng = np.random.RandomState(2026)
rows = []
for it in range(2000):
    tune = np.zeros(len(R), bool)
    for idx in strata.values():
        tune[rng.choice(idx, len(idx) // 2, replace=False)] = True
    best = max(RULES, key=lambda r: prf(tune, r)[2])
    pr, rc, f1 = prf(~tune, best)
    rows.append(dict(rule=f"dHash<={best[0]} & pHash<={best[1]}".replace("<=64", "<=any"), dmax=best[0], pmax=best[1],
                     test_flagged=flagged.get(best), prec_eval=pr, rec_eval=rc, f1_eval=f1))
S = pd.DataFrame(rows)
freq = (S.groupby("rule").agg(veces=("rule", "size"), test_flagged=("test_flagged", "first"),
                              prec_eval=("prec_eval", "mean"), rec_eval=("rec_eval", "mean"))
        .sort_values("veces", ascending=False))
freq["pct"] = 100 * freq.veces / len(S)
freq.round(4).to_csv(OUT / "revision_mitades_reglas.csv")
art = (8, 8)
art_eval = []
rng = np.random.RandomState(7)
for it in range(2000):
    tune = np.zeros(len(R), bool)
    for idx in strata.values():
        tune[rng.choice(idx, len(idx) // 2, replace=False)] = True
    art_eval.append(prf(~tune, art)[:2])
art_eval = np.array(art_eval)
q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanmedian(a)), float(np.nanpercentile(a, 97.5))]
res = dict(iteraciones=len(S), pares=len(R), duplicados_confirmados=int(y.sum()),
           regla_mas_elegida=freq.index[0], pct_mas_elegida=float(freq.pct.iloc[0]),
           pct_elige_regla_articulo=float(100 * ((S.dmax == 8) & (S.pmax == 8)).mean()),
           seleccionada_prec_eval_p2_5_mediana_p97_5=q(S.prec_eval), seleccionada_rec_eval=q(S.rec_eval),
           seleccionada_test_flagged=q(S.test_flagged.astype(float)),
           regla_articulo_prec_eval=q(art_eval[:, 0]), regla_articulo_rec_eval=q(art_eval[:, 1]),
           regla_articulo_test_flagged=flagged[art])
json.dump(res, open(OUT / "revision_mitades_resumen.json", "w"), indent=2)
print(json.dumps(res, indent=1)); print(freq.head(8).round(3))
