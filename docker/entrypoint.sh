#!/bin/sh
# Start a virtual display, then run the command. (xvfb-run hangs as PID 1: it waits for a
# SIGUSR1 from Xvfb, and signals to PID 1 without a handler are dropped.)
Xvfb :99 -screen 0 1280x720x24 -nolisten tcp >/dev/null 2>&1 &
export DISPLAY=:99
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -e /tmp/.X11-unix/X99 ] && break; sleep 0.3; done
exec "$@"
