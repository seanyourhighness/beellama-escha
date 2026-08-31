#!/bin/bash
# Dump the wait channel, syscall and kernel stack of every llama-cli thread.
PID=$(ps aux | grep '[l]lama-cli' | grep -v grep | awk '{print $2}' | head -1)
if [ -z "$PID" ]; then
    echo "no llama-cli process found"
    exit 1
fi
echo "PID=$PID"
echo "wchan: $(cat /proc/$PID/wchan 2>/dev/null)"
echo "syscall: $(cat /proc/$PID/syscall 2>/dev/null)"
echo "State: $(grep State /proc/$PID/status)"
echo "--- thread wait channels ---"
for t in /proc/$PID/task/*; do
    echo "$(basename $t): $(cat $t/wchan 2>/dev/null)"
done | sort | uniq -c | sort -rn | head -12
echo "--- kernel stacks (first 8) ---"
n=0
for t in /proc/$PID/task/*; do
    if [ -r $t/stack ]; then
        echo "== $(basename $t) =="
        head -6 $t/stack
        n=$((n+1))
        [ $n -ge 8 ] && break
    fi
done
