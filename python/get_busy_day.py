from datetime import datetime, timedelta

today = datetime.today()
work_day = today 
for i in range(30):
    day = today + timedelta(days=i)
    if day.weekday() < 5:
        work_day = day
    if day.month != today.month:
        break

print work_day
