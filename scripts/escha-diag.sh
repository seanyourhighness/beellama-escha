#!/bin/bash
PID=$(ps aux | grep '[l]lama-cli -m' | grep -v 'bash' | awk '{print $2}' | head -1)
echo "PID=$PID"
echo "wchan: $(cat /proc/$PID/wchan 2>/dev/null)"
grep -E 'State|VmRSS|Threads' /proc/$PID/status 2>/dev/null
echo "open files: $(ls /proc/$PID/fd 2>/dev/null | wc -l)"
echo "--- thread wait channels ---"
for t in /proc/$PID/task/*; do
    echo "$(basename $t): $(cat $t/wchan 2>/dev/null)"
done | sort | uniq -c | sort -rn | head -10
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
