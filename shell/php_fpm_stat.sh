host=prod
url=localhost/fpm_status

rsp=`curl -s $url`
idle_processes=`echo "$rsp" | grep -E '^idle processes' | grep -Eo "[0-9]+"`
active_processes=`echo "$rsp" | grep -E '^active processes' | grep -Eo "[0-9]+"`
total_processes=`echo "$rsp" | grep -E '^total processes' | grep -Eo "[0-9]+"`
max_active_processes=`echo "$rsp" | grep -E '^max active processes' | grep -Eo "[0-9]+"`
slow_requests=`echo "$rsp" | grep -E '^slow requests' | grep -Eo "[0-9]+"`
accepted_conn=`echo "$rsp" | grep -E '^accepted conn' | grep -Eo "[0-9]+"`

sleep 1
rsp=`curl -s $url`
accepted_conn2=`echo "$rsp" | grep -E '^accepted conn' | grep -Eo "[0-9]+"`
let req_per_sec=accepted_conn2-accepted_conn

rsp=`curl -s "$url?xml&full" | xml2 | grep "/status/processes/process/request-duration" |grep -Eo '[0-9]+$'  `

min_rsp_time=`echo "$rsp" | awk 'NR==1||$0<x{x=$0}END{print x/1000}'`
max_rsp_time=`echo "$rsp" | awk 'NR==1||$0>x{x=$0}END{print x/1000}'`
median_rsp_time=`echo "$rsp" | sort -n|awk '{a[NR]=$0}END{print(NR%2==1)?a[int(NR/2)+1]:(a[NR/2]+a[NR/2+1])/2/1000}'`
avg_rsp_time=`echo "$rsp" | awk '{x+=$0}END{print x/NR/1000}'`

echo "$min_rsp_time $max_rsp_time $median_rsp_time $avg_rsp_time"

echo "php-fpm.$host.idle_processes:$idle_processes|g" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.active_processes:$active_processes|g" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.total_processes:$total_processes|g" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.max_active_processes:$max_active_processes|g" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.slow_requests:$slow_requests|g" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.accepted_conn:$accepted_conn|g" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.req_per_sec:$req_per_sec|g" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.min_rsp_time:$min_rsp_time|ms" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.max_rsp_time:$max_rsp_time|ms" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.median_rsp_time:$median_rsp_time|ms" | nc -u -w0 127.0.0.1 8125
echo "php-fpm.$host.avg_rsp_time:$avg_rsp_time|ms" | nc -u -w0 127.0.0.1 8125
