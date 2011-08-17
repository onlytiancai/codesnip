#encoding=utf-8
'这是一个写的比较烂的函数，求重构'
import httplib
import re

def foo(urls):
    '打印每个网页上.com和.cn域名的个数'
    #抓取网页
    responses = []
    for url in urls:
        conn = httplib.HTTPConnection(url)
        conn.request('get','/')
        response = conn.getresponse().read()
        responses.append(response)

    #获取每个页面的域名
    domains = [] 
    domain_patten = re.compile(r'www\.\w+\.(com|cn)')
    for i in xrange(len(urls)):
        domains.append([])
        for domain in domain_patten.findall(responses[i]):
            domains[i].append(str(domain))

    #计数
    domain_counter = []
    for i in xrange(len(urls)):
        domain_counter.append({})
        for domain in domains[i]:
            if domain_counter[i].has_key(domain):
                domain_counter[i][domain] += 1
            else:
                domain_counter[i][domain] = 1

    #打印结果
    for i in xrange(len(urls)):
        url = urls[i]
        print '*'*20 + url + '*'*20
        counter = domain_counter[i]
        for c in counter:
            print '\t%s-%s' % (c, counter[c])
if __name__ == '__main__':
    urls = ['www.sina.com.cn','sports.sina.com.cn']
    foo(urls)
