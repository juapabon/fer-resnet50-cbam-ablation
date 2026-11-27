#@setup:rafdb
from contextlib import contextmanager

# RAF-DB oficial (subconjunto básico): EmoLabel/list_patition_label.txt + Image/aligned/*_aligned.jpg (100x100).
# También acepta carpetas train/<clase>/ y test/<clase>/ con nombres de clase o con los códigos 1-7 de RAF-DB.
RAF_DIR = Path(os.environ.get("FER_RAFDB_DIR", str(Path.home() / "datasets/RAF-DB")))
RAF_VARIANTS, RAF_SEEDS = ["completo", "base"], CFG["ablation_seeds"]
RUNS_RAF = WORK / "runs_rafdb"
RAF_CODE = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}

def load_rafdb(root):
    lab = sorted(root.rglob("list_patition_label.txt"))
    if lab:
        img_dir = next((d for d in root.rglob("aligned") if d.is_dir()), None)
        assert img_dir is not None, f"No hay carpeta 'aligned' bajo {root}"
        split = {"train": [], "test": []}
        for line in open(lab[0]):
            name, code = line.split()
            p = img_dir / name.replace(".jpg", "_aligned.jpg")
            split["train" if name.startswith("train") else "test"].append((str(p), CLASSES.index(RAF_CODE[int(code)])))
        return split["train"], split["test"]
    def from_folders(d):
        rows = []
        for sub in sorted(Path(d).iterdir()):
            if sub.is_dir():
                c = RAF_CODE[int(sub.name)] if sub.name.isdigit() else sub.name.lower()
                c = {"anger": "angry", "happiness": "happy", "sadness": "sad", "surprised": "surprise"}.get(c, c)
                rows += [(str(f), CLASSES.index(c)) for f in sorted(sub.iterdir()) if f.suffix.lower() in IMG_EXT]
        return rows
    tr = next(d for d in root.rglob("train") if d.is_dir()); te = next(d for d in root.rglob("test") if d.is_dir())
    return from_folders(tr), from_folders(te)

if not 'data_context' in globals():
    @contextmanager
    def data_context(**kw):
        """Sustituye temporalmente los datos globales que usa train_run (train/val/test, runs)."""
        g_ = globals(); old = {k: g_[k] for k in kw}
        g_.update(kw)
        try:
            yield
        finally:
            g_.update(old)

RAF_OK = RAF_DIR.exists()
if RAF_OK:
    raf_train_all, raf_test = load_rafdb(RAF_DIR)
    missing = [p for p, _ in raf_train_all + raf_test if not Path(p).exists()]
    assert not missing, f"{len(missing)} imágenes de la lista no existen, p. ej. {missing[:2]}"
    # validación: 10 % estratificado de train ANTES de balancear (RAF-DB no trae validación), misma semilla
    tr_i, va_i = train_test_split(np.arange(len(raf_train_all)), test_size=CFG["val_fraction"], random_state=CFG["seed"],
                                  stratify=[l for _, l in raf_train_all])
    raf_train_orig = [raf_train_all[i] for i in sorted(tr_i)]; raf_val = [raf_train_all[i] for i in sorted(va_i)]
    raf_gen = build_balanced_train(raf_train_orig, CFG["seed"], tag="_rafdb")
    raf_train = raf_train_orig + raf_gen
    rng_r = np.random.RandomState(CFG["seed"])
    raf_clean = [raf_train_orig[i] for i in sorted(rng_r.choice(len(raf_train_orig), min(CFG["clean_train_eval_n"], len(raf_train_orig)), replace=False))]
    RAF_CTX = dict(train_samples=raf_train, val_samples=raf_val, test_samples=raf_test, train_clean_samples=raf_clean, RUNS_DIR=RUNS_RAF)
    RUNS_RAF.mkdir(exist_ok=True)
    RAF_TABLE = pd.DataFrame({"Class": CLASSES, "Train original": counts(raf_train_orig), "Train generated": counts(raf_gen),
                              "Validation": counts(raf_val), "Test": counts(raf_test)})
    RAF_TABLE.to_csv(OUT / "rafdb_datos.csv", index=False)
    display(RAF_TABLE)
else:
    print(f"RAF-DB no encontrado en {RAF_DIR}: define FER_RAFDB_DIR o copia ahí el subconjunto básico. Se omite la sección 18.")
