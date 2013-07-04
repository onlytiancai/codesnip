# -*- coding: utf-8 -*-

import requests

login_email = 'xxx@xxx.com'
login_password = 'password'

def create_domain(domain):
    data = {'domain': domain,
            'login_email': login_email,
            'login_password': login_password,
            'format': 'json',
            'error_on_empty': 'no'
            }

    r = requests.post('https://dnsapi.cn/Domain.Create', data=data)
    r = r.json()
    print 'create domain', domain, r['status']['code'], r['status']['message']
    return 0 if 'domain' not in r else r['domain']['id']



def create_record(domain, domain_id, sub_domain, value):
    data = {'sub_domain': sub_domain,
            'ttl': 3600,
            'record_type': 'A',
            'value': value,
            'mx': 0,
            'login_email': login_email,
            'login_password': login_password,
            'record_line': u'默认',
            'domain_id': domain_id,
            'domain': domain,
            'format': 'json',
            'error_on_empty': 'no'
            }

    r = requests.post('https://dnsapi.cn/Record.Create', data=data)
    r = r.json()
    print 'create record', domain, sub_domain, r['status']['code'], r['status']['message']

if __name__ == '__main__':
    for line in open('./domain.txt'):
        domain, ip = line.split()
        domain = domain.strip()
        ip = ip.strip()
        domain_id = create_domain(domain) 
        if domain_id != 0:
            for sub_domain in ['www', '*', '*.www']:
                create_record(domain, domain_id, sub_domain, ip)
