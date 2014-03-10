domain=huhao.me
name_server_key='dns9'

ns=`dig ns ${domain} +short`
echo $ns

IFS=' ' read -a dnsservers <<< "${ns}"

for i in "${dnsservers[@]}"
do
    echo "$i"
done
# expr index "${result}" ${name_server_key} 
