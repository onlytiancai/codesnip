#!/bin/bash

PID="$1"
DURATION="${2:-5}"

if [ -z "$PID" ]; then
  echo "Usage: $0 <PID> [SECONDS]"
  exit 1
fi

sed \
  -e "s/__PID__/$PID/g" \
  -e "s/__DURATION__/$DURATION/g" \
  php_stack.bt.tpl \
| sudo bpftrace -