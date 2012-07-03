#!/bin/bash
#network
#Mike.Xu
while : ; do
      time='date +%m"-"%d" "%k":"%M'
      day='date +%m"-"%d'
      rx_before='ifconfig eth0|sed -n "8"p|awk '{print $2}'|cut -c7-'
      tx_before='ifconfig eth0|sed -n "8"p|awk '{print $6}'|cut -c7-'
      sleep 2
      rx_after='ifconfig eth0|sed -n "8"p|awk '{print $2}'|cut -c7-'
      tx_after='ifconfig eth0|sed -n "8"p|awk '{print $6}'|cut -c7-'
      rx_result=$[(rx_after-rx_before)/256]
      tx_result=$[(tx_after-tx_before)/256]
      echo "$time Now_In_Speed: "$rx_result"kbps Now_OUt_Speed: "$tx_result"kbps"
      sleep 2
done

