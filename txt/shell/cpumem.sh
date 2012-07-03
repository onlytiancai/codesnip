#!/bin/bash
#script  to capture system statistics
OUTFILE=/home/xu/capstats.csv
DATE='date +%m/%d/%Y'
TIME='date +%k:%m:%s'
TIMEOUT='uptime'
VMOUT='vmstat 1 2'
 USERS='echo $TIMEOUT | gawk '{print $4}' '
LOAD='echo $TIMEOUT | gawk '{print $9}' | sed "s/,//' '
FREE='echo $VMOUT | sed -n '/[0-9]/p' | sed -n '2p' | gawk '{print $4} ' '
IDLE='echo  $VMOUT | sed -n '/[0-9]/p' | sed -n '2p' |gawk '{print $15}' '
echo "$DATE,$TIME,$USERS,$LOAD,$FREE,$IDLE" >> $OUTFILE

