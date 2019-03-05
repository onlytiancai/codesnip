#-*- coding: utf-8 -*-
import time
import feedparser
from googletrans import Translator

translator = Translator(service_urls=['translate.google.cn'])

url = 'http://feeds.newscientist.com/tech/'
resp = feedparser.parse(url)
for feed in resp['entries'][:10]:

    title = feed['title']
    cn_title= translator.translate(title, src='en', dest='zh-cn').text

    summary = feed['summary']
    cn_summary = translator.translate(summary, src='en', dest='zh-cn').text

    print(time.strftime('%Y-%m-%d %H:%M:%S',feed['published_parsed']))
    print(title)
    print(cn_title)
    print(summary)
    print(cn_summary)
    print('')


