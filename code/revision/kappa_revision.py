"""Punto 3: acuerdo entre anotadores sobre los 120 pares de la hoja ciega.

Uso: python kappa_revision.py <csv exportado por el segundo anotador>
Calcula el kappa de Cohen (3 categorías y binario misma/no misma), el porcentaje de acuerdo y la matriz de
confusión entre anotadores, y la precisión/sensibilidad de la regla del artículo con el veredicto del segundo anotador.
Salida: resultados_frontiers/revision_kappa.json
"""
import json, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

OUT = Path.home() / "TESIS/Articulos Cientifico /resultados_frontiers"
R = pd.read_csv(OUT / "revision_duplicados_muestra.csv", keep_default_na=False).set_index("id")
key = pd.read_csv(OUT / "revision_ciega_120_clave.csv")
a2 = pd.read_csv(sys.argv[1], keep_default_na=False)
df = key.merge(a2, on="hoja_id")
df = df[df.verdict.isin(["same", "different", "uncertain"])]
df["r1"] = R.loc[df.par_id, "verdict"].values
df["d"], df["p"] = R.loc[df.par_id, "dhash_d"].values, R.loc[df.par_id, "phash_d"].values
b1, b2 = df.r1 == "same", df.verdict == "same"
rule = (df.d <= 8) & (df.p <= 8)
res = dict(pares=len(df), anotador=str(df.anotador.iloc[0]) if "anotador" in df and len(df) else "",
           kappa_3_categorias=float(cohen_kappa_score(df.r1, df.verdict, labels=["same", "different", "uncertain"])),
           kappa_binario=float(cohen_kappa_score(b1, b2)), acuerdo_binario_pct=float(100 * (b1 == b2).mean()),
           matriz=pd.crosstab(df.r1, df.verdict).to_dict(),
           regla_articulo_con_anotador2=dict(precision=float((rule & b2).sum() / max(rule.sum(), 1)),
                                             sensibilidad=float((rule & b2).sum() / max(b2.sum(), 1))))
json.dump(res, open(OUT / "revision_kappa.json", "w"), indent=2)
print(json.dumps(res, indent=1))
