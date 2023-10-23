import os
import sqlite3
import logging
import config
import feedparser
from datetime import datetime
from time import mktime
import craw

feed_url = 'https://hnrss.org/newest'

logging.basicConfig(level=logging.DEBUG,
                    handlers=[logging.FileHandler(os.path.basename(__file__) + '.log', 'a', 'utf-8'),
                        logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    )


feeds = feedparser.parse(feed_url)

conn = sqlite3.connect('hn.db')
cursor = conn.cursor()
for feed in feeds.entries:
    try:
        feed.feed_id = int(feed.id.split('=')[-1])
        published = datetime.fromtimestamp(mktime(feed.published_parsed))
        logging.debug('fetch feed:%s', feed.feed_id)

        sql = 'select id from feeds where feed_id=?'
        cursor.execute(sql, (feed.feed_id, ))
        row = cursor.fetchone()
        if not row:
            url, main_content, summary_cn = craw.get_summary(feed.summary[:1024])

            sql = 'insert into feeds(feed_id,author,title,summary,published, url,main_content,summary_cn) values(?, ?, ?, ?, ?, ?, ?, ?)'
            cursor.execute(sql, (feed.feed_id, feed.author[:32], 
                                 feed.title[:512], feed.summary[:1024],
                                 published.strftime('%Y-%m-%d %H:%M:%S'),
                                 url, main_content, summary_cn
                                 ))
            affected = cursor.rowcount
            conn.commit()
            logging.debug('insert feed:%s %s', feed.feed_id, affected)
        else:
            logging.debug('ignore feed:%s', feed.feed_id)
    except Exception as ex:
        error = getattr(ex, 'message', repr(ex))
        logging.error('insert feed error:%s %s', feed.feed_id, error)

conn.close()
