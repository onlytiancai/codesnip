import requests
import hashlib
import time
from pprint import pprint
from datetime import datetime

session = requests.Session()
headers = {
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.80 Safari/537.36',
}
r = session.get(url='https://legulegu.com/stockdata/sw-market-width', headers=headers)
pprint(r.headers)
token = hashlib.md5(datetime.now().strftime('%Y-%m-%d').encode(encoding='utf-8')).hexdigest()
r = session.get('https://legulegu.com/api/stockdata/member-ship/ma-market-width?level=1&severalTradeDays=30&token=%s' % token, headers=headers)
pprint(r.headers)
pprint(r.json())
