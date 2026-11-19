#!/bin/bash
# Cola paralela de los experimentos adicionales del artículo: lanzar_ext.sh dedup|rafdb  (reanudable)
source ~/miniforge3/etc/profile.d/conda.sh; conda activate tesis
G=$1; W=$HOME/fer_frontiers_work; B=$W/cola_$G; NW=${FER_PARALLEL:-4}
export TMPDIR=$W/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg FER_NUM_WORKERS=${FER_NUM_WORKERS:-6}
mkdir -p $B/logs
exec 9>$B/lanzar.lock; flock -n 9 || { echo "ya hay una cola $G en marcha"; exit 0; }
say() { echo "$(date '+%F %T') $*" | tee -a $B/cola.log; }
for ronda in 1 2 3; do
  rm -rf $B/claims
  say "ronda $ronda: preparación"
  python $W/tmp/worker_ext.py $G 0 --prep >> $B/logs/prep.log 2>&1 || { say "fallo en preparación"; exit 1; }
  for i in $(seq 1 $NW); do python $W/tmp/worker_ext.py $G $i >> $B/logs/worker$i.log 2>&1 & sleep 60; done
  wait
  n=$(grep -c "ERROR" $B/logs/worker*.log | awk -F: '{s+=$2} END {print s}')
  say "ronda $ronda terminada (errores acumulados en logs: $n)"
  [ "$(ls -d $B/claims/* 2>/dev/null | wc -l)" -gt 0 ] && [ "$n" = 0 ] && break
done
touch $B/TERMINADO; say "TERMINADO"
