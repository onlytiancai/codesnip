## HTTP Mock by PHP

工作中经常遇到接口联调，但刚开始约定好接口时，接口还没开发完成。
或者依赖第三方的接口，有时候不稳定，这时候就需要一个http mock，
用来做为自己测试的挡板。

现在就用PHP简单实现这个功能。

### 已支持功能

1. 根据路由规则返回指定静态文件的假数据，如json,txt,html,js等；
2. 根据路由规则执行动态的php文件生成假数据，因为有的时候mock也需要少许的逻辑的；
3. 根据路由规则抓取远程url的数据，相当于透明代理，因为有时候不想把所有接口都mock;
4. 如果目标文件不存在，则返回错误；


### 基本使用

    $r = new Route(dirname(__FILE__));

    $r->addRoute("apimock.io", "GET", "/users", "/data/users.json");
    $r->dispatch("apimock.io", "GET", "/users");

    $r->addRoute("apimock.io", "GET", "/users/1", "/get_user.php");
    $r->dispatch("apimock.io", "GET", "/users/1");

    $r->addRoute("apimock.io", "GET", "/fetchweb", "http://ihuhao.com/");
    $r->dispatch("apimock.io", "GET", "/fetchweb");

### 待支持功能

1. 路由规则支持正则匹配
2. 更简洁的，支持目录和tryfile的透明代理模式

### 测试用例

    php test/test.php
