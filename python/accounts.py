# -*- coding: utf-8 -*-

from __future__ import absolute_import
import shelve
import os
import web
import uuid
import hashlib
import json
from datetime import datetime

_curdir = os.path.dirname(__file__)
_session = shelve.open(os.path.join(_curdir, 'session'))
_accounts = shelve.open(os.path.join(_curdir, 'accounts'))


def _save_session(username):
    sessionid = str(uuid.uuid1())
    web.setcookie('sessionid', sessionid, 60 * 60 * 24 * 365)
    _session[sessionid] = web.storage(username=username, createtime=datetime.now())


class login(object):
    def POST(self, username, password):
        password = hashlib.sha1(password).hexdigest()
        if username in _accounts:  # 账户已存在
            account = _accounts[username]
            if account.password != password:      # 登录失败
                return json.dumps(dict(code=401, message='password invalid'))
            else:  # 登录成功
                _save_session(username)
                return json.dumps(dict(code=200, message='login ok'))
        else:  # 账户不存在，自动注册
            _accounts[username] = web.Storage(password=password)
            _save_session(username)
            return json.dumps(dict(code=200, message='register ok'))


class logout(object):
    def POST(self):
        sessionid = web.cookies('sessionid')
        if not sessionid:
            return json.dumps(dict(code=400, message='invalid session'))
        web.setcookie('sessionid', sessionid, -1)
        del _session[sessionid]

        return json.dumps(dict(code=200, message='logout ok'))


class userinfo(object):
    def GET(self):
        sessionid = web.cookies('sessionid')
        if not sessionid:
            return json.dumps(dict(code=400, message='invalid session'))

        username = _session[sessionid]
        account = _accounts[username].copy()
        del account['password']
        return json.dumps(dict(code=200, message='ok', data=account))

    def POST(self):
        sessionid = web.cookies('sessionid')
        if not sessionid:
            return json.dumps(dict(code=400, message='invalid session'))

        username = _session[sessionid]
        account = _accounts[username]
        account.update(web.data())


urls = ["/login", login,
        "/logout", logout,
        "/userinfo", userinfo
        ]


def __del__():
    _session.close()
    _accounts.close()
