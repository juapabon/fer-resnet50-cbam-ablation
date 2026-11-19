# Valores para los [COMPLETAR] del artículo

> Generado el 2026-11-19 01:30

## Entorno y coste (Tabla 3)
- GPU: NVIDIA GeForce RTX 3090 (23.6 GB), driver 570.153.02, CUDA 12.1, cuDNN 90100
- Python 3.8.10, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121; semilla 42; AMP True
- Épocas: 91 (mejor: 76); tiempo de entrenamiento: 1.35 h (53 s/época)
- Parámetros: backbone 23,508,032, CBAM 524,386, cabeza 1,057,799, total 25,090,217

## Datos
- Validación: carpeta validation/; train originales 28704, generadas 21633 (43.0 % del train balanceado); val 3588; test 3595
- Imágenes de test con duplicado en train/val: 312 {'train-test': 383, 'train-val': 442, 'val-test': 63}

## Entrenamiento
- Exactitud de entrenamiento sin mezcla (mejor época): 0.7787; con MixUp: 0.4327; validación: 0.6756

## Test (modelo reentrenado)
- Sin TTA: acc 0.6712 [IC95 0.6557–0.6864], F1 macro 0.6563, F2 macro 0.6424
- Con TTA: acc 0.6801 [IC95 0.6647–0.6952], F1 macro 0.6667, F2 macro 0.6536
- McNemar exacto TTA vs sin TTA: b=112, c=80, p=0.02502
- Bootstrap pareado (B=10000): Δacc=0.0089 [0.0014, 0.0164]; ΔF1=0.0104 [-0.0013, 0.0225]

## Checkpoint publicado (números del artículo)
- Sin TTA: acc 0.6670 [0.6515–0.6823], F1 macro 0.6476
- Con TTA: acc 0.6823 [0.6669–0.6974], F1 macro 0.6726
- McNemar TTA vs sin TTA: p=0.0001086; Δacc bootstrap 0.0153 [0.0078, 0.0231]

## Por clase (TTA)

| Clase | n | Precisión [IC95] | Sensibilidad [IC95] | F1 | F2 | Tipo I | Tipo II |
|---|---|---|---|---|---|---|---|
| angry | 496 | 0.624 [0.581, 0.665] | 0.639 [0.596, 0.680] | 0.631 | 0.636 | 0.062 | 0.361 |
| disgust | 56 | 0.912 [0.770, 0.970] | 0.554 [0.424, 0.676] | 0.689 | 0.601 | 0.001 | 0.446 |
| fear | 513 | 0.571 [0.523, 0.617] | 0.470 [0.427, 0.513] | 0.516 | 0.487 | 0.059 | 0.530 |
| happy | 900 | 0.840 [0.815, 0.862] | 0.887 [0.864, 0.906] | 0.863 | 0.877 | 0.056 | 0.113 |
| neutral | 621 | 0.628 [0.591, 0.663] | 0.684 [0.647, 0.720] | 0.655 | 0.672 | 0.085 | 0.316 |
| sad | 608 | 0.527 [0.488, 0.565] | 0.549 [0.510, 0.588] | 0.538 | 0.545 | 0.100 | 0.451 |
| surprise | 401 | 0.808 [0.765, 0.845] | 0.746 [0.701, 0.786] | 0.776 | 0.757 | 0.022 | 0.254 |

## Ablación (media ± DT)

| Variante | Semillas | Acc TTA | F1 TTA | p Holm |
|---|---|---|---|---|
| completo | 5 | 0.6827 ± 0.0031 | 0.6732 ± 0.0063 | — |
| base | 5 | 0.6880 ± 0.0054 | 0.6850 ± 0.0070 | 0.8308 |
| sin_cbam | 5 | 0.6813 ± 0.0105 | 0.6622 ± 0.0225 | 1 |
| sin_focal | 5 | 0.6899 ± 0.0055 | 0.6846 ± 0.0075 | 0.07462 |
| sin_mixup | 5 | 0.6814 ± 0.0036 | 0.6744 ± 0.0026 | 1 |
| cbam_layer2 | 5 | 0.6869 ± 0.0120 | 0.6756 ± 0.0153 | 0.02962 |
| cbam_layer3 | 5 | 0.6824 ± 0.0088 | 0.6768 ± 0.0107 | 0.532 |
| cbam_layer3_4 | 5 | 0.6763 ± 0.0097 | 0.6670 ± 0.0139 | 1 |
| cbam_layer2_3_4 | 5 | 0.6858 ± 0.0066 | 0.6810 ± 0.0086 | 0.06394 |

## Archivos
- Figuras: `resultados_frontiers/figuras`
- Tablas LaTeX: `resultados_frontiers/tablas`
- Checkpoints y datos generados: `~/fer_frontiers_work`