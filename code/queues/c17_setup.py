#@setup:dedup
from contextlib import contextmanager

# Regla elegida con la revisión manual de la sección 16 (rejilla de 16b): marca un número de imágenes de test
# (647) acorde con la estimación estratificada de duplicados reales, con 1 falso positivo en los 360 pares revisados
DEDUP_RULE = dict(dhash=8, phash=8)
DEDUP_VARIANTS, DEDUP_SEEDS = ["completo", "base"], CFG["ablation_seeds"]
RUNS_DEDUP = WORK / "runs_dedup"

def rule_pairs(hq, hr, same=False, chunk=256):
    """Pares (i, j) con dHash <= umbral y pHash-DCT <= umbral en la misma orientación (normal o especular)."""
    out = []
    for s in range(0, len(hq), chunk):
        e = slice(s, s + chunk); hit = None
        for cd, cp in (("phash", "phash_dct"), ("dhash_flip", "phash_dct_flip")):
            h_ = ((hamming_matrix(hq[cd].values[e], hr.phash.values) <= DEDUP_RULE["dhash"]) &
                  (hamming_matrix(hq[cp].values[e], hr.phash_dct.values) <= DEDUP_RULE["phash"]))
            hit = h_ if hit is None else hit | h_
        if same:
            ii = np.arange(hit.shape[0]); hit[ii, s + ii] = False
        ii, jj = np.nonzero(hit)
        out += list(zip(s + ii, jj))
    return out

def components(n, pairs):
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])

t0 = time.time()
H_valtest = pd.concat([H["val"], H["test"]], ignore_index=True)
P_tr_vt = rule_pairs(H["train"], H_valtest)                 # train con copia en val o test
P_va_te = rule_pairs(H["val"], H["test"])                   # val con copia en test
P_te_tv = rule_pairs(H["test"], H_trval)                    # test con copia en train o val
P_tr_tr = rule_pairs(H["train"], H["train"], same=True)     # repetidas dentro de train
print(f"Pares según la regla en {time.time() - t0:.0f} s")

comp = components(len(H["train"]), P_tr_tr)
lab = H["train"].label.values
cdf = pd.DataFrame({"comp": comp, "label": lab})
n_lab = cdf.groupby("comp").label.nunique()
conflict_comps = set(n_lab[n_lab > 1].index)
first_of_comp = ~pd.Series(comp).duplicated().values
drop_train = set(i for i, _ in P_tr_vt)
drop_within = set(np.where(~first_of_comp)[0])
drop_conflict = set(np.where(np.isin(comp, list(conflict_comps)))[0])
keep_tr = np.array([i not in drop_train | drop_within | drop_conflict for i in range(len(H["train"]))])
drop_val = set(i for i, _ in P_va_te)
TEST_LEAK = set(H["test"].path.iloc[i] for i, _ in P_te_tv)

assert list(H["train"].path) == [p for p, _ in train_orig] and list(H["val"].path) == [p for p, _ in val_samples]
train_orig_dd = [s for s, k in zip(train_orig, keep_tr) if k]
val_dd = [s for i, s in enumerate(val_samples) if i not in drop_val]
train_gen_dd = build_balanced_train(train_orig_dd, CFG["seed"], tag="_dedup")
train_samples_dd = train_orig_dd + train_gen_dd
rng_dd = np.random.RandomState(CFG["seed"])
train_clean_dd = [train_orig_dd[i] for i in sorted(rng_dd.choice(len(train_orig_dd), min(CFG["clean_train_eval_n"], len(train_orig_dd)), replace=False))]
test_clean_mask = np.array([p not in TEST_LEAK for p, _ in test_samples])

DEDUP_SUMMARY = OrderedDict(
    train_original=len(train_orig), removed_dup_with_val_or_test=len(drop_train),
    removed_within_train_repeats=len(drop_within - drop_train),
    removed_label_conflict_clusters=len(drop_conflict - drop_within - drop_train),
    train_after=len(train_orig_dd), val_original=len(val_samples), val_removed_dup_with_test=len(drop_val),
    val_after=len(val_dd), test=len(test_samples), test_with_dup_in_train_or_val=len(TEST_LEAK),
    test_clean=int(test_clean_mask.sum()), train_generated_after=len(train_gen_dd))
json.dump(DEDUP_SUMMARY, open(OUT / "dedup_resumen.json", "w"), indent=2)
pd.DataFrame({"path": sorted(TEST_LEAK)}).to_csv(OUT / "test_con_duplicado_regla.csv", index=False)
pd.DataFrame({"Class": CLASSES, "Train original": counts(train_orig), "Train dedup": counts(train_orig_dd),
              "Val dedup": counts(val_dd), "Test": counts(test_samples),
              "Test with duplicate": counts([s for s in test_samples if s[0] in TEST_LEAK])}).to_csv(OUT / "dedup_por_clase.csv", index=False)

@contextmanager
def data_context(**kw):
    """Sustituye temporalmente los datos globales que usa train_run (train/val/test, runs)."""
    g_ = globals(); old = {k: g_[k] for k in kw}
    g_.update(kw)
    try:
        yield
    finally:
        g_.update(old)

DEDUP_CTX = dict(train_samples=train_samples_dd, val_samples=val_dd, train_clean_samples=train_clean_dd, RUNS_DIR=RUNS_DEDUP)
RUNS_DEDUP.mkdir(exist_ok=True)
pd.Series(DEDUP_SUMMARY)
