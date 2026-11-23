#!/bin/zsh
# Live view of the probe 4 training runs. Run from the AeroNav folder:  zsh experiments/watch_training.sh
# Refreshes every 5 seconds; press Ctrl-C to stop watching (training keeps going in the background).
while true; do
  clear
  echo "AeroNav probe 4 training  ($(date +%H:%M:%S))"
  echo
  echo "Running Python processes (one per drone):"
  ps -o pid,etime,%cpu,rss,command -p $(pgrep -d, -f experiments.middle_training) 2>/dev/null \
    | awk 'NR==1 {print "  PID    TIME   CPU%  MEMORY_MB  DRONE"; next}
           {printf "  %-6s %-6s %-5s %-10d %s %s\n", $1, $2, $3, $4/1024, $(NF-1), $NF}'
  echo
  echo "Checkpoints saved so far (one every 500,000 moves, 10 = finished):"
  for v in plus diamond; do
    for s in 0 1 2; do
      n=$(ls results/models/middle16_${v}_seed${s} 2>/dev/null | wc -l | tr -d ' ')
      bar=$(printf '%*s' $n '' | tr ' ' '#')
      printf "  %-8s seed %s  [%-10s] %2s/10\n" $v $s "$bar" $n
    done
  done
  echo
  echo "Press Ctrl-C to stop watching."
  sleep 5
done
