bash ./service.status.sh
if [ $? -ne 0 ]; then
    echo  `date '+%Y-%m-%d %H:%M'` "service is not available";
    bash ./service.stop.sh
    bash ./service.start.sh
else
    echo  `date '+%Y-%m-%d %H:%M'` "service is available";
fi
