#!/bin/sh
#systemstat.sh
#Mike.Xu
IP=192.168.1.227
top -n 2| grep "Cpu" >>./temp/cpu.txt
free -m | grep "Mem" >> ./temp/mem.txt
df -k | grep "sda1" >> ./temp/drive_sda1.txt
#df -k | grep sda2 >> ./temp/drive_sda2.txt
df -k | grep "/mnt/storage_0" >> ./temp/mnt_storage_0.txt
df -k | grep "/mnt/storage_pic" >> ./temp/mnt_storage_pic.txt
time=`date +%m"."%d" "%k":"%M`
connect=`netstat -na | grep "219.238.148.30:80" | wc -l`
echo "$time  $connect" >> ./temp/connect_count.txt

