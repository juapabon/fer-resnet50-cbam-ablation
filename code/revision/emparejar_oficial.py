"""Reconstruye la partición oficial de FER2013 (Training / PublicTest / PrivateTest) para la copia del artículo.

FERPlus conserva la partición oficial de FER2013 (train = Training, validation = PublicTest, test = PrivateTest).
Cada imagen original de Nuevo_dataset se empareja por píxeles con su imagen de FERPlus: candidatos con dHash
(distancia <= 10) y verificación con el error absoluto medio (MAE)
entre las dos imágenes de 48x48. El emparejamiento es uno a uno (voraz por MAE creciente), así que las copias
idénticas de FER2013 se reparten respetando cuántas hay en cada partición. Se conservan las etiquetas de FER2013.
Salida: ~/fer_frontiers_work/oficial/particion_oficial.csv y resumen_emparejamiento.json
"""
import glob, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

FP = Path.home() / "TESIS/Caras/Nuevos Modelos/dataset/FerPlus/FerPlus"
NU = Path.home() / "TESIS/Caras/Nuevo_dataset"
OUT = Path.home() / "fer_frontiers_work/oficial"; OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SPLIT_FP = {"train": "Training", "validation": "PublicTest", "test": "PrivateTest"}
MAE_MAX = 12.0
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def load(path):
    im = Image.open(path).convert("L")
    if im.size != (48, 48):
        im = im.resize((48, 48), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)

def dhash(a):
    s = np.asarray(Image.fromarray(a).resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if v else "0" for v in (s[:, 1:] > s[:, :-1]).flatten()), 2)

nu_rows = []
for s in ("train", "validation", "test"):
    for k, c in enumerate(CLASSES):
        for f in sorted(glob.glob(str(NU / s / c / "*.jpg"))):
            nu_rows.append((f, s, k))
fp_rows = []
for s in ("train", "validation", "test"):
    for f in sorted(glob.glob(str(FP / s / "*" / "fer*.png"))):
        fp_rows.append((f, SPLIT_FP[s], int(re.findall(r"fer(\d+)", os.path.basename(f))[0])))
print("Nuevo_dataset:", len(nu_rows), "| FERPlus:", len(fp_rows), flush=True)

A = np.stack([load(f) for f, _, _ in nu_rows]); B = np.stack([load(f) for f, _, _ in fp_rows])
ha = np.array([dhash(a) for a in A], dtype=np.uint64); hb = np.array([dhash(b) for b in B], dtype=np.uint64)
Af, Bf = A.reshape(len(A), -1).astype(np.int16), B.reshape(len(B), -1).astype(np.int16)

cand = []   # (mae, i_nuevo, j_ferplus)
for s in range(0, len(ha), 512):
    x = np.bitwise_xor(ha[s:s + 512, None], hb[None, :])
    d = POP[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
    for i, j in zip(*np.nonzero(d <= 10)):
        m = float(np.abs(Af[s + i] - Bf[j]).mean())
        if m <= MAE_MAX:
            cand.append((m, s + i, j))
print("candidatos:", len(cand), flush=True)

cand.sort()
used_a, used_b, match = set(), set(), {}
for m, i, j in cand:
    if i not in used_a and j not in used_b:
        used_a.add(i); used_b.add(j); match[i] = (j, m)

# ambiguas: imágenes de Nuevo con candidatos casi tan buenos como el elegido (MAE <= mejor + 2) en otra partición
best = {}
for m, i, j in cand:
    best.setdefault(i, m)
splits_of = {}
for m, i, j in cand:
    if m <= best[i] + 2.0:
        splits_of.setdefault(i, set()).add(fp_rows[j][1])
rows = []
for i, (f, s, k) in enumerate(nu_rows):
    j, m = match.get(i, (None, np.nan))
    rows.append(dict(path=f, label=k, split_copia=s, split_oficial=fp_rows[j][1] if j is not None else "",
                     ferplus_idx=fp_rows[j][2] if j is not None else -1, mae=m,
                     ambigua=len(splits_of.get(i, ())) > 1))
df = pd.DataFrame(rows)
df.to_csv(OUT / "particion_oficial.csv", index=False)
res = dict(nuevo=len(nu_rows), ferplus=len(fp_rows), emparejadas=int((df.split_oficial != "").sum()),
           sin_pareja=int((df.split_oficial == "").sum()), ambiguas_entre_particiones=int(df.ambigua.sum()),
           mae_mediana=float(df.mae.median()), mae_p99=float(df.mae.quantile(0.99)), mae_max=float(df.mae.max()),
           ferplus_sin_pareja=len(fp_rows) - len(used_b),
           cruce=pd.crosstab(df.split_copia, df.split_oficial.replace("", "sin pareja")).to_dict(),
           por_clase=pd.crosstab(df.label.map(dict(enumerate(CLASSES))), df.split_oficial.replace("", "sin pareja")).to_dict())
json.dump(res, open(OUT / "resumen_emparejamiento.json", "w"), indent=2)
print(json.dumps(res, indent=1))
