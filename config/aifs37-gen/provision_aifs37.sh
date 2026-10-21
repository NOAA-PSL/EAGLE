#!/bin/bash
# Compose, realize and provision the aifs37 run directories (A0/A2 on MI300X, A1 on H100, smoke tests
# 0a = A1 recipe on 1 H100, 0b = A0 recipe on 1 MI300X). Rewrites each run dir's training.yaml and
# runscript (provisioning skips files that exist) and copies the prebuilt O96/o48 graph; leaves logs and
# checkpoints. Usage, from EAGLE/: config/aifs37-gen/provision_aifs37.sh [a0 a1 a2 smoke-0a smoke-0b]
set -euo pipefail
TMP=$(mktemp -d)
GRAPH=run/aifs2-1deg-smoke/training/outputs/graph.pt
declare -A COMPOSE=(
  [a0]=base:aifs2-1deg:aifs37:aifs37-a0:ursa:anemoi-dev:mi300x
  [a1]=base:aifs2-1deg:aifs37:aifs37-a1:ursa:anemoi-dev
  [a2]=base:aifs2-1deg:aifs37:aifs37-a2:ursa:anemoi-dev:mi300x:gsienkf-gpu # second MI300X run on its own account
  [smoke-0a]=base:aifs2-1deg:aifs37:aifs37-a1:ursa:anemoi-dev:aifs2-1deg-smoke:aifs37-smoke
  [smoke-0b]=base:aifs2-1deg:aifs37:aifs37-a0:ursa:anemoi-dev:aifs2-1deg-smoke:aifs37-smoke:mi300x:mi300x-1gpu
)
declare -A RUNDIR=([a0]=aifs37-a0 [a1]=aifs37-a1 [a2]=aifs37-a2 [smoke-0a]=aifs37-smoke [smoke-0b]=aifs37-smoke-0b)
for r in ${@:-a0 a1 a2 smoke-0a smoke-0b}; do
  make -s config compose=${COMPOSE[$r]} > $TMP/$r.yaml 2>/dev/null
  [[ $r == smoke-0b ]] && sed -i '0,/experiment_name: aifs37-smoke$/s//experiment_name: aifs37-smoke-0b/' $TMP/$r.yaml
  make -s realize config=$TMP/$r.yaml update=update-aifs37.yaml > eagle-aifs37-$r.yaml 2>/dev/null
  d=run/${RUNDIR[$r]}/training
  rm -f $d/training.yaml $d/runscript.training
  ANEMOI_ENV=anemoi-dev make -s training config=eagle-aifs37-$r.yaml task=provisioned_rundir 2>&1 | tail -n 1
  # chained resume jobs (sbatch -d afterany) exit once the run has finished
  sed -i 's/^time srun/[ -f runscript.training.done ] \&\& { echo "run already finished"; exit 0; }\ntime srun/' $d/runscript.training
  # submit them as `sbatch runscript.training resume`: they never start a fresh run (if the first job was
  # cancelled, afterany would otherwise start one, e.g. on data that failed its patch)
  sed -i 's/^time srun/[ "${1:-}" = resume ] \&\& ! ls outputs\/checkpoint\/*\/last.ckpt >\/dev\/null 2>\&1 \&\& { echo "resume job but no checkpoint"; exit 1; }\ntime srun/' $d/runscript.training
  # the 1-GPU MI300X smoke test shares the node
  [[ $r == smoke-0b ]] && sed -i '/^#SBATCH --exclusive$/d' $d/runscript.training
  mkdir -p $d/outputs; [[ -f $d/outputs/graph.pt ]] || cp $GRAPH $d/outputs/
done
rm -rf $TMP
