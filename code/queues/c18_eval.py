# RAF-DB: completo frente a base (media ± DT, t pareada, McNemar del conjunto) y evaluación cruzada entre conjuntos
# (modelos de FER-2013 sobre el test de RAF-DB y al revés), con la misma TTA de 3 vistas.
@torch.no_grad()
def cross_eval(run_dir, spec, samples):
    m = ResNet50CBAM(NUM_CLASSES, tuple(spec["cbam"]), pretrained=False).to(device)
    m.load_state_dict(torch.load(Path(run_dir) / "best.pth", map_location=device, weights_only=True))
    _, pt, _ = predict(m, make_loader(samples, None, False, tta=True), tta=True)
    del m; torch.cuda.empty_cache()
    return pt

RAF_RES = RAF_CMP = RAF_CROSS = None
test_samples_fer, RUNS_FER = test_samples, RUNS_DIR
if RAF_OK:
    with data_context(**RAF_CTX):
        y_raf = labels_of(raf_test)
        rp = {v: runs_preds(RUNS_RAF, v, RAF_SEEDS) for v in RAF_VARIANTS}
        rows = []
        for v in RAF_VARIANTS:
            for tta in (False, True):
                a, f, _ = summarize(rp[v], np.ones(len(raf_test), bool), tta)
                rows.append(dict(variant=v, tta=tta, n_seeds=len(a), acc_mean=a.mean(), acc_sd=a.std(ddof=1) if len(a) > 1 else np.nan,
                                 f1_mean=f.mean(), f1_sd=f.std(ddof=1) if len(f) > 1 else np.nan))
        RAF_RES = pd.DataFrame(rows)
        common = sorted(set(rp["completo"]) & set(rp["base"]))
        if len(common) >= 2:
            all_ = np.ones(len(raf_test), bool)
            ac, fc, ec = summarize({s: rp["completo"][s] for s in common}, all_)
            ab, fb, eb = summarize({s: rp["base"][s] for s in common}, all_)
            RAF_CMP = dict(seeds=len(common), d_acc=(ac - ab).mean(), t_p_acc=stats.ttest_rel(ac, ab).pvalue, d_f1=(fc - fb).mean(),
                           t_p_f1=stats.ttest_rel(fc, fb).pvalue, mcnemar_ens_p=mcnemar_exact(ec == y_raf, eb == y_raf)["p_value"],
                           ens_acc_completo=(ec == y_raf).mean(), ens_f1_completo=fast_macro_f1(y_raf, ec))
            RAF_PC = per_class_table(y_raf, ec); RAF_PC.to_csv(OUT / "rafdb_por_clase_ensemble_completo.csv", index=False)
            display(RAF_PC.round(3))
        # evaluación cruzada: sólo 'completo', semillas comunes
        cross = []
        for s in RAF_SEEDS:
            fer_run, raf_run = RUNS_FER / f"completo_s{s}", RUNS_RAF / f"completo_s{s}"
            if (fer_run / "meta.json").exists():
                pt = cross_eval(fer_run, VARIANTS["completo"], raf_test)
                cross.append(dict(train="FER-2013", test="RAF-DB", seed=s, acc=(pt.argmax(1) == y_raf).mean(), f1=fast_macro_f1(y_raf, pt.argmax(1))))
            if (raf_run / "meta.json").exists():
                pt = cross_eval(raf_run, VARIANTS["completo"], test_samples_fer)
                cross.append(dict(train="RAF-DB", test="FER-2013", seed=s, acc=(pt.argmax(1) == y_test).mean(), f1=fast_macro_f1(y_test, pt.argmax(1))))
        RAF_CROSS = pd.DataFrame(cross)
    if len(RAF_CROSS):
        RAF_CROSS.to_csv(OUT / "cruzado_fer_rafdb_por_semilla.csv", index=False)
        display(RAF_CROSS.groupby(["train", "test"])[["acc", "f1"]].agg(["mean", "std"]).round(4))
    RAF_RES.to_csv(OUT / "rafdb_resultados.csv", index=False)
    json.dump(RAF_CMP, open(OUT / "rafdb_completo_vs_base.json", "w"), indent=2, default=float)
    display(RAF_RES.round(4)); print(RAF_CMP)

# Auditoría de duplicados en RAF-DB con la misma regla de la sección 17 (test frente a train+val)
RAF_DUP = None
if RAF_OK:
    HR = {k: extra_hashes(hash_table(sm, f"rafdb_{k}"), f"rafdb_{k}") for k, sm in (("trainval", raf_train_orig + raf_val), ("test", raf_test))}
    raf_leak = sorted(set(i for i, _ in rule_pairs(HR["test"], HR["trainval"])))
    RAF_DUP = dict(test_with_dup_in_trainval=len(raf_leak), pct=100 * len(raf_leak) / len(raf_test),
                   md5_exact=int(HR["test"].md5.isin(set(HR["trainval"].md5)).sum()))
    json.dump(RAF_DUP, open(OUT / "rafdb_duplicados.json", "w"), indent=2)
    print("RAF-DB, duplicados test-train/val:", RAF_DUP)
