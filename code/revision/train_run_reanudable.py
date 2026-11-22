# train_run reanudable (misma lógica que la celda 22 del cuaderno). Al final de cada época guarda last.pth
# (modelo, optimizador, scheduler, scaler, historial y estados aleatorios) y, si existe, continúa desde ahí.
# best.pth se escribe de forma atómica y sólo después de guardar last.pth, para que siempre case con el historial.
# El lock de la carpeta de la run lo toma el worker antes de llamar.
import os as _os, random as _random

def _rng_state(loader):
    return dict(py=_random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loader=loader.generator.get_state() if loader.generator is not None else None)

def _set_rng_state(st, loader):
    _random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    if st["loader"] is not None and loader.generator is not None:
        loader.generator.set_state(st["loader"])

def train_run_reanudable(variant, seed, verbose=True):
    run_dir = RUNS_DIR / f"{variant}_s{seed}"
    meta_f, last_f = run_dir / "meta.json", run_dir / "last.pth"
    if meta_f.exists():
        return json.load(open(meta_f))
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = VARIANTS[variant]
    set_seed(seed)
    model = ResNet50CBAM(NUM_CLASSES, spec["cbam"], CFG["pretrained"]).to(device)
    criterion = make_criterion(spec["loss"])
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    train_loader = make_loader(train_samples, train_transform, True, seed, drop_last=True)
    val_loader = make_loader(val_samples, eval_transform, False)
    clean_loader = make_loader(train_clean_samples, eval_transform, False)
    y_val, y_clean = labels_of(val_samples), labels_of(train_clean_samples)

    hist, best, bad, best_epoch, start, t_prev, resumed = [], -1.0, 0, -1, 0, 0.0, []
    stage_starts = {s["start_epoch"]: s for s in CFG["stages"]}
    opt = sch = None
    if last_f.exists():
        ck = torch.load(last_f, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt, sch = apply_stage(model, stage_starts[max(e for e in stage_starts if e <= ck["epoch"] - 1)])
        opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"]); scaler.load_state_dict(ck["scaler"])
        hist, best, bad, best_epoch = ck["hist"], ck["best"], ck["bad"], ck["best_epoch"]
        start, t_prev, resumed = ck["epoch"], ck["train_time_s"], ck["resumed"] + [ck["epoch"]]
        _set_rng_state(ck["rng"], train_loader)
        pend = run_dir / f"best_pendiente_e{best_epoch}.pth"
        if pend.exists():
            _os.replace(pend, run_dir / "best.pth")
        print(f"  reanudo {variant} s{seed} tras la época {start} (mejor {best_epoch}, F1 val {best:.4f})", flush=True)
        del ck
    for f in run_dir.glob("best_pendiente_e*.pth"):
        f.unlink()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(start, CFG["max_epochs"] if bad < CFG["patience"] else start):
        if epoch in stage_starts:
            opt, sch = apply_stage(model, stage_starts[epoch])
            if verbose:
                print(f"  época {epoch + 1}: etapa {stage_starts[epoch]['train']}")
        t0 = time.time()
        tr_loss, tr_acc_mix = train_one_epoch(model, train_loader, criterion, opt, scaler, spec["mixup"])
        pc, _, _ = predict(model, clean_loader)
        pv, _, va_loss = predict(model, val_loader, criterion)
        va_pred = pv.argmax(1)
        va_f1 = f1_score(y_val, va_pred, average="macro", zero_division=0)
        sch.step(va_f1)
        row = dict(epoch=epoch + 1, train_loss=tr_loss, train_acc_mixup=tr_acc_mix,
                   train_acc_clean=float((pc.argmax(1) == y_clean).mean()), val_loss=va_loss,
                   val_acc=accuracy_score(y_val, va_pred), val_f1_macro=va_f1,
                   lrs={pg["name"]: pg["lr"] for pg in opt.param_groups}, epoch_time_s=time.time() - t0)
        hist.append(row)
        improved = va_f1 > best + CFG["min_delta"]
        if improved:
            best, bad, best_epoch = va_f1, 0, epoch + 1
            torch.save(model.state_dict(), run_dir / f"best_pendiente_e{best_epoch}.pth")
        else:
            bad += 1
        if verbose:
            print(f"  [{epoch + 1:3d}] loss {tr_loss:.4f}/{va_loss:.4f} | acc mix {tr_acc_mix:.3f} limpio "
                  f"{row['train_acc_clean']:.3f} val {row['val_acc']:.3f} | F1 val {va_f1:.4f}"
                  f"{' *' if improved else ''} | {row['epoch_time_s']:.0f}s", flush=True)
        pd.DataFrame(hist).to_json(run_dir / "history.json", orient="records", indent=1)
        torch.save(dict(epoch=epoch + 1, model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(),
                        scaler=scaler.state_dict(), hist=hist, best=best, bad=bad, best_epoch=best_epoch,
                        train_time_s=t_prev + time.time() - t_start, rng=_rng_state(train_loader), resumed=resumed),
                   run_dir / "last.pth.tmp")
        _os.replace(run_dir / "last.pth.tmp", last_f)
        if improved:
            _os.replace(run_dir / f"best_pendiente_e{best_epoch}.pth", run_dir / "best.pth")
        if bad >= CFG["patience"]:
            break
    train_time = t_prev + time.time() - t_start

    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device, weights_only=True))
    t0 = time.time()
    p0, pt, _ = predict(model, make_loader(test_samples, None, False, tta=True), tta=True)
    infer_time = time.time() - t0
    save_test_predictions(run_dir / "preds_test.csv", p0, pt)
    meta = dict(variant=variant, seed=seed, spec={k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()},
                epochs_run=len(hist), best_epoch=best_epoch, best_val_f1=best, train_time_s=train_time,
                sec_per_epoch=train_time / len(hist), test_infer_time_tta_s=infer_time,
                peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0,
                trainable_params_final=n_params([p for p in model.parameters() if p.requires_grad]),
                run_dir=str(run_dir), reanudada_tras_epocas=resumed)
    json.dump(meta, open(meta_f, "w"), indent=2)
    last_f.unlink(missing_ok=True)
    del model; torch.cuda.empty_cache()
    return meta
