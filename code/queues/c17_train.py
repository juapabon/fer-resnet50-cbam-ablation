# Entrenamiento sin repetidas: 'completo' y 'base' x 5 semillas en WORK_DIR/runs_dedup (reanudable: reutiliza lo terminado).
# En el servidor se lanzaron en paralelo con worker_ext.py; aquí sólo se reutilizan.
RUN_DEDUP = True
if RUN_DEDUP:
    with data_context(**DEDUP_CTX):
        for v in DEDUP_VARIANTS:
            for s in DEDUP_SEEDS:
                train_run(v, s, verbose=False)
