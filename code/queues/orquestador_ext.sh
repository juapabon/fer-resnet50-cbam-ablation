#!/bin/bash
# Orquestador de los experimentos adicionales del artículo Frontiers. Lo lanza cron cada 30 min y @reboot.
# Idempotente y reanudable: cada entrenamiento terminado (meta.json) se reutiliza; uno interrumpido vuelve a empezar.
#   1) cola sin repetidas (runs_dedup)  2) cola RAF-DB si los datos existen  3) cuaderno final ejecutado
W=$HOME/fer_frontiers_work; T=$W/tmp; D="$HOME/TESIS/Articulos Cientifico "
RAF=${FER_RAFDB_DIR:-$HOME/TESIS/datasets_emociones/RAF-DB}
L=$W/orquestador.log
exec 8>$W/orquestador.lock; flock -n 8 || exit 0
say() { echo "$(date '+%F %T') $*" >> $L; }
done_runs() { ls $W/$1/*/meta.json 2>/dev/null | wc -l; }
cola_viva() { [ -e $W/cola_$1/lanzar.lock ] && ! flock -n $W/cola_$1/lanzar.lock true 2>/dev/null; }

if [ "$(done_runs runs_dedup)" -lt 10 ]; then
  cola_viva dedup || { say "relanzo cola dedup ($(done_runs runs_dedup)/10 hechas)"; rm -f $W/cola_dedup/TERMINADO; setsid nohup $T/lanzar_ext.sh dedup >/dev/null 2>&1 < /dev/null & }
  exit 0
fi
if [ -d "$RAF" ] && [ "$(done_runs runs_rafdb)" -lt 10 ]; then
  cola_viva rafdb || { say "relanzo cola rafdb ($(done_runs runs_rafdb)/10 hechas)"; rm -f $W/cola_rafdb/TERMINADO; FER_RAFDB_DIR="$RAF" setsid nohup $T/lanzar_ext.sh rafdb >/dev/null 2>&1 < /dev/null & }
  exit 0
fi
# 2b) experimentos de la revisión (worker_rev.py), cuando RAF-DB y su cuaderno ya terminaron
if [ -e $W/CUADERNO_OK_rafdb ] && [ ! -e $W/cola_rev/TERMINADO ]; then
  cola_viva rev || { say "relanzo cola rev ($(python3 -c 'print()' >/dev/null; ls $W/runs_{ajuste,oficial,muestreo,semillas,ajuste_fino}/*/meta.json 2>/dev/null | wc -l) runs hechas)"; setsid nohup $T/rev/lanzar_rev.sh >/dev/null 2>&1 < /dev/null & }
  exit 0
fi
# 2c) experimentos del informe Q1 (worker_rev2.py), cuando la cola rev ya terminó
if [ -e $W/cola_rev/TERMINADO ] && [ ! -e $W/cola_rev2/TERMINADO ]; then
  cola_viva rev2 || { say "relanzo cola rev2 ($(ls $W/runs_{ft_ablacion,sin_balanceo,rafdb_color,convnext}/*/meta.json 2>/dev/null | wc -l) runs hechas fuera de runs_oficial)"; setsid nohup $T/rev/lanzar_rev2.sh >/dev/null 2>&1 < /dev/null & }
  exit 0
fi
cola_viva dedup && exit 0; cola_viva rafdb && exit 0
etapa=dedup; [ -d "$RAF" ] && etapa=rafdb
[ -e $W/CUADERNO_OK_$etapa ] && exit 0
say "cuaderno final ($etapa)"
source ~/miniforge3/etc/profile.d/conda.sh; conda activate tesis
export TMPDIR=$T MPLBACKEND=Agg FER_RAFDB_DIR="$RAF"
if (cd "$D" && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
      FER_ResNet50_CBAM_Frontiers_ablacion.ipynb --output FER_ResNet50_CBAM_Frontiers_completo_ejecutado.ipynb) > $W/cuaderno_final_$etapa.log 2>&1; then
  touch $W/CUADERNO_OK_$etapa; say "TERMINADO $etapa: cuaderno ejecutado y resultados en resultados_frontiers/"
else
  say "fallo del cuaderno final: $(grep -a Error $W/cuaderno_final_$etapa.log | tail -1)"
fi
