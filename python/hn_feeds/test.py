from cyac import AC

import base64
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

import sys
print(match_with_ac(sys.argv[1]))
