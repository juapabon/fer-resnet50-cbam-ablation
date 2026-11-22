#!/bin/bash
# Cola de los experimentos de la revisión (worker_rev.py): una sola cola con prioridades, reanudable.
# Cada run se reserva con flock y guarda last.pth en cada época: tras una caída o un reinicio continúa donde iba.
source ~/miniforge3/etc/profile.d/conda.sh; conda activate tesis
W=$HOME/fer_frontiers_work; B=$W/cola_rev; NW=${FER_PARALLEL:-4}
export TMPDIR=$W/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg FER_NUM_WORKERS=${FER_NUM_WORKERS:-6}
mkdir -p $B/logs
exec 9>$B/lanzar.lock; flock -n 9 || { echo "ya hay una cola rev en marcha"; exit 0; }
say() { echo "$(date '+%F %T') $*" | tee -a $B/cola.log; }
# relevo_pids ("hueco pid" por línea): workers anteriores que aún terminan su run; su hueco se ocupa cuando acaban
vivo() { [ -n "$1" ] && grep -qa worker_rev /proc/$1/cmdline 2>/dev/null; }
for ronda in 1 2 3 4 5 6; do
  say "ronda $ronda: preparación"
  python $W/tmp/rev/worker_rev.py 0 --prep >> $B/logs/prep.log 2>&1 || { say "fallo en preparación"; exit 1; }
  for i in $(seq 1 $NW); do
    old=$(awk -v i=$i '$1==i {print $2}' $B/relevo_pids 2>/dev/null)
    ( while vivo $old; do sleep 60; done; exec python $W/tmp/rev/worker_rev.py $i >> $B/logs/worker$i.log 2>&1 ) &
    sleep 60
  done
  wait
  rm -f $B/relevo_pids
  p=$(python $W/tmp/rev/worker_rev.py 0 --pending 2>/dev/null | tail -1)
  say "ronda $ronda terminada (trabajos pendientes: $p)"
  [ "$p" = 0 ] && break
done
[ "$p" = 0 ] && { touch $B/TERMINADO; say "TERMINADO"; }
