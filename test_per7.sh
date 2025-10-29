#!/bin/bash
# Test script with timeout
python per7_format.py &
PID=$!
sleep 15
kill $PID 2>/dev/null
wait $PID 2>/dev/null
