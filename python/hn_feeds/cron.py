import os
import pymysql
import logging
import config
import feedparser

conn = pymysql.connect(db='hn', cursorclass=pymysql.cursors.DictCursor, **config.conn)
feed_url = 'https://hnrss.org/newest'

logging.basicConfig(level=logging.DEBUG,
                    handlers=[logging.FileHandler(os.path.basename(__file__) + '.log', 'a', 'utf-8'),
                        logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    )


with conn.cursor() as cursor:
    feeds = feedparser.parse(feed_url)
    for feed in feeds.entries:
        feed.feed_id = feed.id.split('=')[-1]
        logging.debug('fetch feed:%s', feed.id)
        try:
            sql = 'select id from feeds where feed_id=%s'
            cursor.execute(sql, feed.feed_id)
            row = cursor.fetchone()
            if not row:
                sql = 'insert into feeds values(null, %s,%s,%s,%s,%s)'
                affected = cursor.execute(sql, (feed.feed_id, feed.author[:32], 
                                                feed.title[:512], feed.summary[:1024],
                                                feed.published_parsed))
                conn.commit()
                logging.debug('insert feed:%s %s', feed.feed_id, affected)
            else:
                logging.debug('ignore feed:%s', feed.feed_id)
        except Exception as ex:
            error = getattr(ex, 'message', repr(ex))
            logging.error('insert feed error:%s %s', feed.feed_id, error)
