def runs_preds(runs_dir, variant, seeds):
    out = {}
    for s in seeds:
        d = Path(runs_dir) / f"{variant}_s{s}"
        if (d / "meta.json").exists():
            out[s] = load_test_predictions(d)
    return out

def summarize(preds, mask, tta=True):
    accs, f1s, P = [], [], []
    for s, (y, p0, pt) in preds.items():
        p = (pt if tta else p0)
        a = p.argmax(1)
        accs.append((a[mask] == y[mask]).mean()); f1s.append(fast_macro_f1(y[mask], a[mask])); P.append(p)
    return np.array(accs), np.array(f1s), (np.mean(P, 0).argmax(1) if P else None)

DEDUP_ROWS, DEDUP_TESTS = [], []
subsets = {"full test": np.ones(len(test_samples), bool), "clean test": test_clean_mask}
for v in DEDUP_VARIANTS:
    orig, dd = runs_preds(RUNS_DIR, v, DEDUP_SEEDS), runs_preds(RUNS_DEDUP, v, DEDUP_SEEDS)
    common = sorted(set(orig) & set(dd))
    for sub, m in subsets.items():
        for name, pr in (("original", orig), ("deduplicated", dd)):
            a, f, _ = summarize(pr, m)
            DEDUP_ROWS.append(dict(variant=v, train_data=name, test=sub, n_seeds=len(a), n_test=int(m.sum()),
                                   acc_mean=a.mean() if len(a) else np.nan, acc_sd=a.std(ddof=1) if len(a) > 1 else np.nan,
                                   f1_mean=f.mean() if len(f) else np.nan, f1_sd=f.std(ddof=1) if len(f) > 1 else np.nan))
        if len(common) >= 2:
            ao, fo, eo = summarize({s: orig[s] for s in common}, m)
            ad, fd, ed = summarize({s: dd[s] for s in common}, m)
            y_ = y_test[m]
            DEDUP_TESTS.append(dict(variant=v, test=sub, seeds=len(common), d_acc=(ad - ao).mean(), d_f1=(fd - fo).mean(),
                                    t_p_acc=stats.ttest_rel(ad, ao).pvalue, t_p_f1=stats.ttest_rel(fd, fo).pvalue,
                                    mcnemar_ens_p=mcnemar_exact(ed[m] == y_, eo[m] == y_)["p_value"]))
DEDUP_RES = pd.DataFrame(DEDUP_ROWS); DEDUP_CMP = pd.DataFrame(DEDUP_TESTS)
DEDUP_RES.to_csv(OUT / "dedup_resultados.csv", index=False); DEDUP_CMP.to_csv(OUT / "dedup_contrastes.csv", index=False)

# completo frente a base después de depurar (¿se mantiene la conclusión de la ablación?)
cb = []
for sub, m in subsets.items():
    pc, pb = runs_preds(RUNS_DEDUP, "completo", DEDUP_SEEDS), runs_preds(RUNS_DEDUP, "base", DEDUP_SEEDS)
    common = sorted(set(pc) & set(pb))
    if len(common) >= 2:
        ac, fc, ec = summarize({s: pc[s] for s in common}, m); ab, fb, eb = summarize({s: pb[s] for s in common}, m)
        cb.append(dict(test=sub, seeds=len(common), d_acc_completo_minus_base=(ac - ab).mean(), t_p_acc=stats.ttest_rel(ac, ab).pvalue,
                       d_f1=(fc - fb).mean(), t_p_f1=stats.ttest_rel(fc, fb).pvalue,
                       mcnemar_ens_p=mcnemar_exact(ec[m] == y_test[m], eb[m] == y_test[m])["p_value"]))
DEDUP_CB = pd.DataFrame(cb); DEDUP_CB.to_csv(OUT / "dedup_completo_vs_base.csv", index=False)
display(DEDUP_RES.round(4)); display(DEDUP_CMP.round(4)); display(DEDUP_CB.round(4))
