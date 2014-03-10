### 精简版的web.py

抽取了web.py的核心代码，只有100多行，能完成web.py最主要url路由，处理请求，返回应答，设置header等功能。

1. 适合写一些纯RESTFull api的网站，不带界面。
1. 只有100多行代码，一眼就能看的清清楚楚，而且只使用了python自带的类库，不会存在不可知的因素，排查问题非常方便。
1. 为了提高性能，可以用gunicorn去跑app.wsgifunc()。

去掉了一些功能。

1. 去掉了模版相关功能，可以自己选择自己喜欢的jinja2,mustache等模版。
1. 去掉了session相关功能，这年头估计没人用session了吧，直接把状态存redis吧。
1. 去掉了db相关的功能，还是直接用sqlalchemy吧。
1. 去掉了form相关的功能，感觉没啥大用。
1. 正则的性能优化逻辑去掉了，有些复杂。

相关链接

1. [Set a cookie and retrieve it with Python and WSGI](http://stackoverflow.com/questions/14107260/set-a-cookie-and-retrieve-it-with-python-and-wsgi)
1. [How can i access cookies directly from WSGI environment](http://stackoverflow.com/questions/9608145/how-can-i-access-cookies-directly-from-wsgi-environment)
1. [Parsing the Request - Get](http://webpython.codepoint.net/wsgi_request_parsing_get)
1. [Parsing the Request - Post](http://webpython.codepoint.net/wsgi_request_parsing_post)
