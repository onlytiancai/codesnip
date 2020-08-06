list='192.168.1.2 192.178.1.3 192.168.1.3'
for ip in $list;do
    out=$(mysql --connect-timeout 1 -h $ip -uroot -ppassword -Bne "select version();" 2>/dev/null)
    if [ ! -z "$out" ];then
        echo $ip ok
    else
        echo $ip no ok
    fi
done
