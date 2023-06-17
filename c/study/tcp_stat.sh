format="%-10s %-10s %-10s %-10s %-10s %-10s %-10s\n"; 
i=0;
while true;do
    time=`date +%H:%m:%S`
    if (( $i%20 == 0 ))
    then
        printf "$format" "time" "SYN-SENT" "ESTAB" "FIN-WAIT-1" "CLOSE-WAIT" "LAST-ACK" "UNCONN";
    fi
    ss | awk -v format="$format" -v time="$time" '{a[$2]+=1} END {printf format, time, a["SYN-SENT"]?a["SYN-SENT"]:0, a["ESTAB"]?a["ESTAB"]:0,a["FIN-WAIT-1"]?a["FIN-WAIT-1"]:0, a["CLOSE-WAIT"]?a["CLOSE-WAIT"]:0,a["LAST-ACK"]?a["LAST-ACK"]:0, a["UNCONN"]?a["UNCONN"]:0}'
    let i+=1;
    sleep 1;
done
