# Entrenamiento en RAF-DB con exactamente el mismo protocolo e hiperparámetros que en FER-2013 (sin reajustar nada).
# En el servidor se lanzó en paralelo con worker_ext.py; aquí sólo se reutiliza lo terminado.
if RAF_OK:
    with data_context(**RAF_CTX):
        for v in RAF_VARIANTS:
            for s in RAF_SEEDS:
                train_run(v, s, verbose=False)
