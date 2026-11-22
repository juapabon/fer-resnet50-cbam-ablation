# Resultados de la revisión manual: precisión por banda (Wilson 95 %), estimación estratificada del número real de
# imágenes de test con duplicado en train/val y precisión/sensibilidad estimadas de varias reglas de decisión.
REVIEW = pd.read_csv(OUT / "revision_duplicados_muestra.csv", keep_default_na=False)
assert REVIEW.verdict.isin(["same", "different", "uncertain"]).all(), "Faltan veredictos en revision_duplicados_muestra.csv"
REVIEW["is_dup"] = REVIEW.verdict == "same"          # 'uncertain' cuenta como no duplicado (estimación conservadora)

band_rows = []
for kind, smp in (("d", "dHash"), ("p", "pHash")):
    dist = NN[("test", "train+val", kind)][0]
    for lo, hi in REVIEW_BANDS[kind]:
        b = f"{lo}" if lo == hi else f"{lo}-{hi}"
        g_ = REVIEW[(REVIEW["sample"] == smp) & (REVIEW.band == b)]
        k, n, N = int(g_.is_dup.sum()), len(g_), int(((dist >= lo) & (dist <= hi)).sum())
        lo_ci, hi_ci = wilson(k, n)
        band_rows.append(dict(hash=smp, band=b, test_images_in_band=N, reviewed=n, same=k, uncertain=int((g_.verdict == "uncertain").sum()),
                              precision=k / n, ci_low=lo_ci, ci_high=hi_ci, est_true_dups=N * k / n,
                              label_conflicts_among_same=int((g_.is_dup & (g_.label_test != g_.label_ref)).sum())))
DUP_REVIEW = pd.DataFrame(band_rows)
DUP_REVIEW.to_csv(OUT / "revision_duplicados_por_banda.csv", index=False)
display(DUP_REVIEW.round(3))

# peso de cada par revisado = tamaño de su banda / pares revisados en la banda (estimador estratificado)
REVIEW["w"] = [DUP_REVIEW.set_index(["hash", "band"]).loc[(r["sample"], r.band)].pipe(lambda x: x.test_images_in_band / x.reviewed)
               for _, r in REVIEW.iterrows()]
RULES = OrderedDict([
    ("dHash = 0 (Section 5)", lambda d, p: d <= 0),
    ("dHash <= 1", lambda d, p: d <= 1),
    ("dHash <= 4", lambda d, p: d <= 4),
    ("pHash <= 4", lambda d, p: p <= 4),
    ("pHash <= 6", lambda d, p: p <= 6),
    ("pHash <= 8", lambda d, p: p <= 8),
    ("dHash <= 8 and pHash <= 8", lambda d, p: (d <= 8) & (p <= 8)),
    ("dHash <= 10 and pHash <= 12", lambda d, p: (d <= 10) & (p <= 12)),
])
rule_rows = []
for name, f in RULES.items():
    row = dict(rule=name)
    for smp in ("dHash", "pHash"):
        r_ = REVIEW[REVIEW["sample"] == smp]; flag = f(r_.dhash_d.values, r_.phash_d.values); y_ = r_.is_dup.values; w = r_.w.values
        tp = (w * flag * y_).sum()
        row[f"precision_{smp}_sample"] = tp / (w * flag).sum() if flag.any() else float("nan")
        row[f"recall_{smp}_sample"] = tp / (w * y_).sum()
    rule_rows.append(row)
DUP_RULES = pd.DataFrame(rule_rows)
DUP_RULES.to_csv(OUT / "revision_duplicados_reglas.csv", index=False)
same = REVIEW[REVIEW.is_dup]
print(f"Pares confirmados: {len(same)}; con etiqueta distinta entre las dos copias: "
      f"{(same.label_test != same.label_ref).sum()} ({100 * (same.label_test != same.label_ref).mean():.1f} %); "
      f"especulares: {same.mirrored.sum()}")
for smp in ("dHash", "pHash"):
    print(f"Estimación (muestra {smp}) de imágenes de test con duplicado real en train/val: "
          f"{DUP_REVIEW[DUP_REVIEW.hash == smp].est_true_dups.sum():.0f} de {len(H['test'])}")
display(DUP_RULES.round(3))

# Rejilla de reglas conjuntas: imágenes de test marcadas con CUALQUIER pareja en train+val (no sólo el vecino más cercano)
# frente a los aciertos/fallos sobre los 360 pares revisados. Una regla útil marca un número de imágenes cercano a la
# estimación estratificada anterior y apenas tiene falsos positivos en la muestra.
q_, r_ = H["test"], H_trval
_D = [hamming_matrix(q_[c].values, r_.phash.values).astype(np.uint8) for c in ("phash", "dhash_flip")]
_P = [hamming_matrix(q_[c].values, r_.phash_dct.values).astype(np.uint8) for c in ("phash_dct", "phash_dct_flip")]
grid = []
for td in (0, 2, 4, 6, 8, 10, 64):
    for tp in (0, 4, 6, 8, 10, 12, 64):
        if td == tp == 64:
            continue
        hit = (((_D[0] <= td) & (_P[0] <= tp)) | ((_D[1] <= td) & (_P[1] <= tp))).any(1)
        f = (REVIEW.dhash_d <= td) & (REVIEW.phash_d <= tp)
        grid.append(dict(dhash_max=td, phash_max=tp, test_flagged=int(hit.sum()), reviewed_flagged=int(f.sum()),
                         reviewed_TP=int((f & REVIEW.is_dup).sum()), reviewed_FP=int((f & ~REVIEW.is_dup).sum()),
                         reviewed_FN=int((~f & REVIEW.is_dup).sum())))
del _D, _P
DUP_GRID = pd.DataFrame(grid).replace({64: "-"})
DUP_GRID.to_csv(OUT / "duplicados_rejilla_reglas.csv", index=False)
DUP_GRID

# Fig. 9: proporción de duplicados reales por banda de distancia (IC de Wilson) y ejemplos revisados
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, smp, col in zip(axes, ("dHash", "pHash"), ("#2b6cb0", "#dd6b20")):
    t_ = DUP_REVIEW[DUP_REVIEW.hash == smp].copy()
    t_["lo_"] = t_.band.str.split("-").str[0].astype(int); t_ = t_.sort_values("lo_")
    x_ = np.arange(len(t_))
    ax.errorbar(x_, t_.precision, yerr=[t_.precision - t_.ci_low, t_.ci_high - t_.precision], fmt="o-", color=col, capsize=3)
    ax.set_xticks(x_); ax.set_xticklabels(t_.band, fontsize=8); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(f"{smp if smp == 'dHash' else 'pHash-DCT'} distance (bits)")
    for xi, n in zip(x_, t_.test_images_in_band):
        ax.annotate(f"{n}", (xi, 1.0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6, color="#718096")
axes[0].set_ylabel("Proportion of true duplicates\n(manual review, 20 pairs/band)")
fig.suptitle("Test image vs. nearest train/val image: proportion of true duplicates per distance band\n(grey numbers: test images in the band)", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"fig9_revision_duplicados.{ext}", dpi=300, bbox_inches="tight")
plt.show(); plt.close(fig)
