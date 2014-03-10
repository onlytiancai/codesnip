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


class LoginException(Exception):
    pass


def _save_session(username):
    sessionid = str(uuid.uuid1())
    web.setcookie('sessionid', sessionid, 60 * 60 * 24 * 365)
    _session[sessionid] = web.storage(username=username, createtime=datetime.now())


def _account_exists(username):
    return username in _accounts


def _check_login(username, password):
    account = _accounts[username]
    if account.password != password:
        return dict(code=401, message='password invalid')
    else:
        _save_session(username)
        return dict(code=200, message='login ok')


def _register_account(username, password):
    _accounts[username] = web.Storage(password=password)
    _save_session(username)
    return json.dumps(dict(code=200, message='register ok'))


def _check_session():
    sessionid = web.cookies('sessionid')
    if not sessionid:
        raise LoginException('sessionid not found in cookies')
    if sessionid not in _session:
        raise LoginException('sessionid not found in sessions')
    return sessionid, _session[sessionid]


class login(object):
    def POST(self, username, password):
        password = hashlib.sha1(password).hexdigest()
        if _account_exists(username):
            result = _check_login(username, password)
        else:
            result = _register_account(username, password)
        return json.dumps(result)


class logout(object):
    def POST(self):
        try:
            sessionid, username = _check_session()
            web.setcookie('sessionid', sessionid, -1)
            del _session[sessionid]
            return json.dumps(dict(code=200, message='logout ok'))
        except LoginException, le:
            return json.dumps(dict(code=400, message=le.message))


class userinfo(object):
    def GET(self):
        try:
            sessionid, username = _check_session()
            account = _accounts[username].copy()
            del account['password']
            return json.dumps(dict(code=200, message='ok', data=account))
        except LoginException, le:
            return json.dumps(dict(code=400, message=le.message))

    def POST(self):
        try:
            sessionid, username = _check_session()
            account = _accounts[username]
            account.update(web.data())
        except LoginException, le:
            return json.dumps(dict(code=400, message=le.message))


urls = ["/login", login,
        "/logout", logout,
        "/userinfo", userinfo
        ]


def __del__():
    _session.close()
    _accounts.close()
