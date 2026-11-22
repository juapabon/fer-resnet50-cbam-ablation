# Muestras estratificadas por banda de distancia (una por hash): cada imagen de test junto a su vecino más cercano
# en train+val según ese hash. Se guardan láminas para la revisión visual y un CSV con la columna 'verdict' vacía.
REVIEW_BANDS = {"d": [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8), (9, 10), (11, 12)],
                "p": [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12), (14, 14), (16, 18), (0, 0)]}
REV_DIR = FIG_DIR / "revision_duplicados"; REV_DIR.mkdir(exist_ok=True)

def review_sample(kind, seed):
    dist, arg, mirr = NN[("test", "train+val", kind)]
    rng = np.random.RandomState(seed); rows = []
    for lo, hi in REVIEW_BANDS[kind]:
        idx = np.where((dist >= lo) & (dist <= hi))[0]
        for i in sorted(rng.choice(idx, min(REVIEW_PER_BAND, len(idx)), replace=False)):
            q, r = H["test"].iloc[i], H_trval.iloc[arg[i]]
            rows.append(dict(sample=f"{kind}Hash", band=f"{lo}" if lo == hi else f"{lo}-{hi}", test_path=q.path,
                             ref_path=r.path, ref_split=r.split, label_test=CLASSES[q.label], label_ref=CLASSES[r.label],
                             dhash_d=pair_dist(q, r, "d"), phash_d=pair_dist(q, r, "p"), mirrored=bool(mirr[i]),
                             md5_equal=q.md5 == r.md5, verdict=""))
    return rows

def review_sheet(grp, title, path, cols=5):
    rows = math.ceil(len(grp) / cols)
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(2 * cols * 1.5, rows * 1.85), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for j, r in enumerate(grp.itertuples()):
        ax_a, ax_b = axes[j // cols, 2 * (j % cols)], axes[j // cols, 2 * (j % cols) + 1]
        ax_a.imshow(Image.open(r.test_path).convert("L"), cmap="gray", vmin=0, vmax=255)
        ax_b.imshow(Image.open(r.ref_path).convert("L"), cmap="gray", vmin=0, vmax=255)
        ax_a.set_title(f"#{r.id} {r.label_test[:3]}", fontsize=7)
        ax_b.set_title(f"{r.label_ref[:3]} d{r.dhash_d}/p{r.phash_d}{' M' if r.mirrored else ''}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

REVIEW = pd.DataFrame(review_sample("d", CFG["seed"]) + review_sample("p", CFG["seed"] + 1))
REVIEW.insert(0, "id", range(len(REVIEW)))
rev_csv = OUT / "revision_duplicados_muestra.csv"
if rev_csv.exists():   # conserva los veredictos ya anotados a mano (por par de imágenes)
    old = pd.read_csv(rev_csv, keep_default_na=False).set_index(["sample", "test_path", "ref_path"])
    for c in ("verdict", "reviewer"):
        if c in old:
            REVIEW[c] = [old[c].get((r["sample"], r.test_path, r.ref_path), "") for _, r in REVIEW.iterrows()]
REVIEW.to_csv(rev_csv, index=False)
for (smp, band), grp in REVIEW.groupby(["sample", "band"], sort=False):
    review_sheet(grp, f"{smp} band {band}: test (left) vs nearest train/val image (right); d/p = dHash/pHash distance, M = mirrored",
                 REV_DIR / f"{smp}_banda_{band}.png")
print(f"{len(REVIEW)} pares en {rev_csv}; láminas en {REV_DIR}")
