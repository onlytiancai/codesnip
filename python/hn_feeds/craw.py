import os
import re
import sqlite3
import logging
import config
import requests
from fatgoose3 import FatGoose
import fake_useragent
from pprint import pprint



logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.basename(__file__) + '.log', 'a', 'utf-8'),
                        logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    )
logging.getLogger("requests").setLevel(logging.WARNING)

reg = re.compile(r'article url.*<a\s+href="([^"]+)"', re.IGNORECASE)
fake_ua = fake_useragent.UserAgent()
goose = FatGoose()

db = sqlite3.connect('hn.db')
db.row_factory = sqlite3.Row
cursor = db.cursor()
sql = 'select id,summary from feeds order by id desc limit 10'
cursor.execute(sql, ())
rows = cursor.fetchall()
for row in rows:
    url = reg.search(row['summary']).group(1)
    print(row['id'], url)
    if url:
        try:
            resp = requests.get(url, headers={'User-Agent': fake_ua.random}, verify=False, timeout=10)
            news = goose.extract(url=url, raw_html=resp.text)
            main_content = news.cleaned_text
            if main_content:
                print(main_content)
        except Exception as ex:
            logging.info('craw error:%s', ex)

db.close()
