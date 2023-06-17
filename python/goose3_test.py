import requests
from fatgoose3 import FatGoose
import fake_useragent

ua = fake_useragent.UserAgent()
g = FatGoose()
import sys
url = 'https://pubs.aip.org/aip/ape/article/1/1/016107/2884934/A-multifunctional-highway-system-incorporating'
if len(sys.argv) > 1:
    url = sys.argv[1]
resp = requests.get(url, headers={'User-Agent': ua.random})
news = g.extract(url=url, raw_html=resp.text)

print(f'news.title: %s\n' % news.title)
print(f'news.author: %s\n' % news.authors)
print(f'news.publish_date: %s\n' % news.publish_date)
print(f'news.cleaned_text: %s\n' % news.cleaned_text)
print(f'news.infos: %s\n' % news.infos)
g.close()
