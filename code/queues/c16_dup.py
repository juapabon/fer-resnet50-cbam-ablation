#@setup:dup
from scipy.fftpack import dct as _dct

DUP_THRESHOLDS = [0, 1, 2, 3, 4, 6, 8, 10, 12]
REVIEW_PER_BAND = 20

def _bits(b):
    return int("".join("1" if v else "0" for v in b.flatten()), 2)

def dhash_img(im):
    a = np.asarray(im.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return _bits(a[:, 1:] > a[:, :-1])

def dct_phash_img(im):
    """pHash clásico: DCT 2D de la imagen 32x32; bits de los 8x8 coeficientes de baja frecuencia frente a su mediana."""
    a = np.asarray(im.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = _dct(_dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return _bits(d > np.median(d))

def extra_hashes(h, name):
    """Añade dHash de la imagen volteada y pHash-DCT (normal y volteada). Caché en hex en WORK_DIR."""
    cols = ["dhash_flip", "phash_dct", "phash_dct_flip"]
    cache = WORK / f"hashes_extra_{name}.csv"
    if cache.exists() and not SMOKE:
        e = pd.read_csv(cache, dtype=str)
        if list(e.path) == list(h.path):
            for c in cols:
                h[c] = np.array([int(v, 16) for v in e[c]], dtype=np.uint64)
            return h
    vals = {c: [] for c in cols}
    for p in h.path:
        im = Image.open(p).convert("L"); fl = im.transpose(Image.FLIP_LEFT_RIGHT)
        vals["dhash_flip"].append(dhash_img(fl)); vals["phash_dct"].append(dct_phash_img(im))
        vals["phash_dct_flip"].append(dct_phash_img(fl))
    for c in cols:
        h[c] = np.array(vals[c], dtype=np.uint64)
    pd.DataFrame({"path": h.path, **{c: [f"{int(v):016x}" for v in h[c]] for c in cols}}).to_csv(cache, index=False)
    return h

def hamming_matrix(a, b):
    x = np.bitwise_xor(np.asarray(a, dtype=np.uint64)[:, None], np.asarray(b, dtype=np.uint64)[None, :])
    return POPCOUNT[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)

def hash_cols(kind):
    return ("phash", "dhash_flip") if kind == "d" else ("phash_dct", "phash_dct_flip")

def nearest(hq, hr, kind, same=False, chunk=256):
    """Distancia mínima (y vecino) de cada imagen de hq a hr, considerando también la versión especular.
    kind='d' (dHash, columna 'phash' de la sección 5) o 'p' (pHash-DCT). same=True excluye la propia imagen."""
    c, cf = hash_cols(kind)
    q, qf, r = hq[c].values, hq[cf].values, hr[c].values
    dmin, arg, mirr = np.empty(len(q), int), np.empty(len(q), int), np.empty(len(q), bool)
    for s in range(0, len(q), chunk):
        d0, d1 = hamming_matrix(q[s:s + chunk], r), hamming_matrix(qf[s:s + chunk], r)
        if same:
            ii = np.arange(d0.shape[0]); d0[ii, s + ii] = 99; d1[ii, s + ii] = 99
        d = np.minimum(d0, d1)
        a = d.argmin(1); ii = np.arange(len(a))
        dmin[s:s + chunk], arg[s:s + chunk], mirr[s:s + chunk] = d[ii, a], a, d1[ii, a] < d0[ii, a]
    return dmin, arg, mirr

def pair_dist(ra, rb, kind):
    c, cf = hash_cols(kind)
    return int(min(hamming_matrix([ra[c]], [rb[c]])[0, 0], hamming_matrix([ra[cf]], [rb[c]])[0, 0]))

t0 = time.time()
for k in H:
    H[k] = extra_hashes(H[k], k)
H_trval = pd.concat([H["train"].assign(split="train"), H["val"].assign(split="val")], ignore_index=True)

NN = {}   # (consulta, referencia, hash) -> (distancia, índice del vecino, especular)
for kind in ("d", "p"):
    NN[("test", "train+val", kind)] = nearest(H["test"], H_trval, kind)
    NN[("val", "train", kind)] = nearest(H["val"], H["train"], kind)
    NN[("val", "test", kind)] = nearest(H["val"], H["test"], kind)
    NN[("train", "train", kind)] = nearest(H["train"], H["train"], kind, same=True)
print(f"Hashes extra y vecinos más cercanos en {time.time() - t0:.0f} s")

sens_rows = []
for (qn, rn, kind), (d, _, m) in NN.items():
    for t in DUP_THRESHOLDS:
        hit = d <= t
        sens_rows.append(dict(query=qn, reference=rn, hash="dHash" if kind == "d" else "pHash-DCT", threshold=t,
                              images_with_match=int(hit.sum()), pct=100 * hit.mean(), mirrored=int((hit & m).sum())))
DUP_SENS = pd.DataFrame(sens_rows)
DUP_SENS.to_csv(OUT / "duplicados_sensibilidad_umbral.csv", index=False)
display(DUP_SENS.pivot_table(index=["hash", "threshold"], columns=["query", "reference"], values="images_with_match"))
