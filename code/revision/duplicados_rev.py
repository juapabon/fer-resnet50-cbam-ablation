"""Duplicados para el análisis de la revisión, con la misma regla conjunta del artículo (cuaderno, sección 16):
una imagen de consulta queda marcada si alguna imagen de referencia cumple dHash <= td y pHash-DCT <= tp,
en la orientación original o en la especular.

  - Partición oficial: PrivateTest frente a Training + PublicTest, td = tp = 8  -> oficial_test_con_duplicado_regla.csv
  - Sensibilidad: test de la copia frente a train + val, td = tp = 10 (887 imágenes) -> test_con_duplicado_regla10.csv
    (y comprobación de que td = tp = 8 vuelve a dar las 647 de test_con_duplicado_regla.csv)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

W = Path.home() / "fer_frontiers_work"
OUT = Path.home() / "TESIS" / "Articulos Cientifico " / "resultados_frontiers"
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def load_hashes():
    """dHash (columna 'phash' de hashes_*_d.csv), dHash especular y pHash-DCT (normal y especular) de todas las imágenes."""
    out = []
    for s in ("train", "val", "test"):
        d = pd.read_csv(W / f"hashes_{s}_d.csv", dtype={"phash": str})
        e = pd.read_csv(W / f"hashes_extra_{s}.csv", dtype=str)
        assert list(d.path) == list(e.path)
        out.append(pd.DataFrame(dict(path=d.path, split=s, dhash=[np.uint64(int(v)) for v in d.phash],
                                     **{c: [np.uint64(int(v, 16)) for v in e[c]] for c in ("dhash_flip", "phash_dct", "phash_dct_flip")})))
    return pd.concat(out, ignore_index=True)

def ham(q, r):
    """Distancias de Hamming (uint8) entre los vectores uint64 q (n) y r (m)."""
    x = np.bitwise_xor(q[:, None], r[None, :])
    return POP[x.view(np.uint8)].reshape(len(q), len(r), 8).sum(2, dtype=np.uint8)

def flagged(Q, R, td, tp, chunk=256):
    rd, rp = R.dhash.values.astype(np.uint64), R.phash_dct.values.astype(np.uint64)
    hit = np.zeros(len(Q), bool)
    for i in range(0, len(Q), chunk):
        q = Q.iloc[i:i + chunk]
        a = (ham(q.dhash.values.astype(np.uint64), rd) <= td) & (ham(q.phash_dct.values.astype(np.uint64), rp) <= tp)
        b = (ham(q.dhash_flip.values.astype(np.uint64), rd) <= td) & (ham(q.phash_dct_flip.values.astype(np.uint64), rp) <= tp)
        hit[i:i + chunk] = (a | b).any(1)
    return hit

H = load_hashes()
res = {}
# sensibilidad en la partición de la copia
Q, R = H[H.split == "test"], H[H.split != "test"]
f8 = flagged(Q, R, 8, 8); f10 = flagged(Q, R, 10, 10)
prev = set(pd.read_csv(OUT / "test_con_duplicado_regla.csv").path)
res.update(copia_test=len(Q), copia_regla8=int(f8.sum()), copia_regla8_igual_articulo=set(Q.path[f8]) == prev,
           copia_regla10=int(f10.sum()))
pd.DataFrame(dict(path=Q.path[f10])).to_csv(OUT / "test_con_duplicado_regla10.csv", index=False)

# partición oficial
po = pd.read_csv(W / "oficial" / "particion_oficial.csv", keep_default_na=False)
sp = dict(zip(po.path, po.split_oficial))
H["oficial"] = H.path.map(sp).fillna("")
Q, R = H[H.oficial == "PrivateTest"], H[H.oficial.isin(["Training", "PublicTest"])]
fo = flagged(Q, R, 8, 8)
fo_tr = flagged(Q, H[H.oficial == "Training"], 8, 8)
res.update(oficial_privatetest=len(Q), oficial_ref=len(R), oficial_regla8=int(fo.sum()), oficial_regla8_solo_training=int(fo_tr.sum()),
           oficial_privatetest_limpio=int(len(Q) - fo.sum()))
pd.DataFrame(dict(path=Q.path[fo])).to_csv(OUT / "oficial_test_con_duplicado_regla.csv", index=False)
json.dump(res, open(OUT / "rev_duplicados.json", "w"), indent=2)
print(json.dumps(res, indent=2))
