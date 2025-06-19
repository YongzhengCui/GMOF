#!/bin/bash
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done
  
  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}


gpus=(0)
threads=1
times=2
config=gmof
env_configs=(sc2_gen_terran) 
notes="for_test_"
# run parallel
count=0
for env_config in "${env_configs[@]}"; do
    name="$config""_""$env_config""_10_11_""$notes"
    for((i=0;i<times;i++)); do
        gpu=${gpus[$(($count % ${#gpus[@]}))]}  
        CUDA_VISIBLE_DEVICES="$gpu" python3 src/main.py --config="$config" with name="$name" --env-config=$env_config "${args[@]}" &

        count=$(($count + 1)) 
        if [ $(($count % $threads)) -eq 0 ]; then
            wait
        fi
        # for random seeds
        sleep $((RANDOM % 10 + 10))
    done
done
wait
