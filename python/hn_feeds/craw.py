import os
import re
import sqlite3
import logging
import config
import requests
from fatgoose3 import FatGoose
import fake_useragent
from pprint import pprint
import json
import time
from tldextract import extract
from cyac import AC
import base64


logging.getLogger("requests").setLevel(logging.WARNING)

reg = re.compile(r'article url.*<a\s+href="([^"]+)"', re.IGNORECASE)
fake_ua = fake_useragent.UserAgent()
goose = FatGoose()
keywords = set()
for line in open('pub_banned_words.txt'):
    keyword = base64.b64decode(line).decode('utf-8').strip()
    keywords.add(keyword)
keywords = list(keywords)
ac_keywords = AC.build(keywords)

def match_with_ac(description):
    results = set()
    for id, start, end in ac_keywords.match(description.lower()):
        results.add(keywords[id])
    return results

def summary_openai(text, trycount=1):
    url = 'http://127.0.0.1:3000/v1/chat/completions'
    headers = {'Content-Type':'application/json', 'Authorization': 'Bearer '+ config.OPENAI_API_KEY}

    message = '请将如下英文返回300字的中文摘要。\n'
    message += text[:20480]

    data = json.dumps({'model':'gpt-3.5-turbo', "messages": [{"role": "user", "content": message}]})
    r = None 
    time_start = time.time()
    try:
        r = requests.post(url, data, headers=headers, timeout=100)
        logging.info('recv resp from openai: %s %s', r.status_code, r.text)
        r = r.json()
    except Exception as ex:
        logging.info('call openai error:%s', ex)
        
    time_end = time.time()
    if not r:
        if time_end - time_start > 10 and trycount < 3:
            return summary_openai(text, trycount+1)
        else:
            return None

    if 'error' in r: 
        return None

    return r['choices'][0]['message']['content'].strip()

def get_main_content(url):
    tsd, td, tsu = extract(url)
    domain = td + '.' + tsu
    if domain in ['youtube.com']:
       return None 
    try:
        resp = requests.get(url, headers={'User-Agent': fake_ua.random}, verify=False, timeout=10)
        news = goose.extract(url=url, raw_html=resp.text)
        return news.cleaned_text
    except Exception as ex:
        logging.info('craw error:%s', ex)


def get_summary(text):
    url = reg.search(text).group(1)
    if not url:
        return '', '', ''
    main_content = get_main_content(url)
    if not main_content:
        return url, '', ''
    summary = summary_openai(main_content)
    if not summary:
        return url, main_content, ''
    if match_with_ac(summary):
        return url, 'xxx', ''
    return url, main_content, summary

if __name__ == '__main__':
    print(get_summary('''
<div class="media-body">
          <h4 class="media-heading">Help there's a dead CEO in my head</h4>
          <p>Article URL: <a href="https://www.experimental-history.com/p/help-theres-a-dead-ceo-in-my-head">https://www.experimental-history.com/p/help-theres-a-dead-ceo-in-my-head</a></p>
<p>Comments URL: <a href="https://news.ycombinator.com/item?id=35722591">https://news.ycombinator.com/item?id=35722591</a></p>
<p>Points: 1</p>
<p># Comments: 0</p>
          <p class="text-right">paulpauper at 2023-04-27 02:27:52</p>
        </div>
    '''))
