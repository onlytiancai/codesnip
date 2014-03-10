# -*- coding: utf-8 -*-

import web
import json
import urllib
import urllib2
import logging

app_jslink = '<script src="/static/sea-modules/seajs/1.3.0/sea-debug.js" data-main="/dnspod/static/main"></script>'
app_desc = 'DNSPod'


class Api(object):
    __headers = {"Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/json",
        "User-Agent": "onlytiancai oauth/0.0.1 (onlytiancai@gmail.com)"}

    def POST(self, api):
        userinfo = web.app_extensions.get_userinfo()
        if userinfo['data']['usertype'] != 'dnspod':
            result = {'status':dict(code=400, message='bad request')}
            result = json.dumps(result)
            return result

        uri = 'https://www.dnspod.cn/Api/' + api
        data = dict(web.input())
        data['access_token'] = userinfo['data']['password']
        data['format'] = 'json'
        data['lang'] = 'cn'
        for key in data:
            data[key] = data[key].encode('utf-8')
        logging.debug('dnspod api:%s %s', uri, data)
        request = urllib2.Request(uri, urllib.urlencode(data), headers=self.__headers)
        response = urllib2.urlopen(request)
        return response


urls = ["/Api/(.*)", Api,
        ]

app = web.application(urls, globals())

if __name__ == "__main__":
    app.run()
