#!/bin/bash
# Cola de los experimentos del informe Q1 (worker_rev2.py): una sola cola con prioridades, reanudable.
# Cada run se reserva con flock y guarda last.pth en cada época: tras una caída o un reinicio continúa donde iba.
# La relanza el orquestador (cron cada 30 min y @reboot) mientras no exista cola_rev2/TERMINADO.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate tesis
W=$HOME/fer_frontiers_work; B=$W/cola_rev2; NW=${FER_PARALLEL:-4}
export TMPDIR=$W/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg FER_NUM_WORKERS=${FER_NUM_WORKERS:-6}
mkdir -p $B/logs
exec 9>$B/lanzar.lock; flock -n 9 || { echo "ya hay una cola rev2 en marcha"; exit 0; }
say() { echo "$(date '+%F %T') $*" | tee -a $B/cola.log; }
for ronda in 1 2 3 4 5 6; do
  say "ronda $ronda: preparación"
  python $W/tmp/rev/worker_rev2.py 0 --prep >> $B/logs/prep.log 2>&1 || { say "fallo en preparación"; exit 1; }
  for i in $(seq 1 $NW); do
    python $W/tmp/rev/worker_rev2.py $i >> $B/logs/worker$i.log 2>&1 &
    sleep 60
  done
  wait
  p=$(python $W/tmp/rev/worker_rev2.py 0 --pending 2>/dev/null | tail -1)
  say "ronda $ronda terminada (trabajos pendientes: $p)"
  [ "$p" = 0 ] && break
done
[ "$p" = 0 ] && { touch $B/TERMINADO; say "TERMINADO"; }
