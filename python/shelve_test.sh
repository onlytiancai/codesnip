#!/bin/bash

curdir=$(cd "$(dirname $0)"; pwd)

# kill -9
rm d -f

for i in {1..5} 
do
    echo $i
    python ${curdir}/shelve_test.py "write"  &
    sleep 2
    ps -ef | grep ${curdir}/shelve_test.py | grep -v grep | cut -c 9-15 | xargs kill -9
done

python ${curdir}/shelve_test.py "read"
