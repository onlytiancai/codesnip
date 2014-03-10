# 第一个javascript 程序

写一个javascript应用不难，但写一个规范的javascript应用不简单，要使用好一些事实标准的模块和工具。

## 一次性安装工具

有一些工具需要提前安装好，所有的项目都会用到

### 安装nodejs

虽然我很讨厌nodejs，但它已经无处不在了.

    git clone git://github.com/creationix/nvm.git nvm
    . ./nvm/nvm.sh
    nvm install v0.8.10
    nvm use v0.8.10

### 安装grunt

yemman用grunt不奇怪，现在连jquery等都用grunt了，说明grunt在社区里已经很强大了，可以自动进行jslint,单元测试，压缩等任务。

    npm install -g grunt

### 安装h5bp

估计很少有人不使用它了。

    npm install https://github.com/h5bp/node-build-script/tarball/master -g

### 安装spm

前端包管理虽然有jamjs，但还是要支持一下国产的spm.

    npm install spm -g

### 安装phantom

单元测试和UI测试需要它，毫不犹豫的安装吧。

    cd ~/tmp
    wget http://phantomjs.googlecode.com/files/phantomjs-1.7.0-linux-x86_64.tar.bz2
    tar xf phantomjs-1.7.0-linux-x86_64.tar.bz2
    mkdir ~/.phantomjs
    mv phantomjs-1.7.0-linux-x86_64/* ~/.phantomjs/
    echo "export PATH=~/.phantomjs/bin/:\${PATH}" >> ~/.bashrc
    . ~/.bashrc

## 开始一个项目

一次性工具装好了，可以开始写一个新项目了,就叫jsempty吧。

### 初始化目录

用h5bp初始化项目，会自动生成相关文件。

    mkdir jsempty
    cd jsempty
    h5bp init

### 初始化依赖的包

    #install seajs, jquery, 必选
    cd jsempty/js
    spm install seajs
    spm install jquery

    #install underscore, backbone，可选
    cd jsempty/js
    spm install underscore
    spm install backbone

    #install bootstrap, 可选，see also https://gist.github.com/1422879
    cd jsempty
    wget http://twitter.github.com/bootstrap/assets/bootstrap.zip
    unzip bootstrap.zip
    rm bootstrap.zip
    cp bootstrap/js/* js/
    cp bootstrap/img/* img/
    cp bootstrap/css/* css/
    rm bootstrap/ -rf

### 开发js模块

上面装好了别人写好的js模块，自己肯定也要开发的，就在js目录下的source里写吧。

    cd jsempty/js
    mkdir sources
    cd sources
    mkdir hello
    cd hello

用grunt的模板初始化模块目录,目前我发现jquery是最适合前端模块的，期待有一天能grunt init:seajs

    grunt init:jquery 

实现一个小功能，vi src/hello.js


    /*global define*/

    define(function(require, exports) {
      exports.hello = function(name) {
          return 'hello ' + name;
        };
    });

因为我们要用国产的seaJS，所以修改下自动生成的测试页面，vi test/hello.html

    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Pyempty Hello Test Suite</title>
      <link rel="stylesheet" href="../libs/qunit/qunit.css" media="screen">
      <script src="../libs/qunit/qunit.js"></script>
      <script src="../../../seajs/1.2.1/sea.js"></script>
      <script src="hello_test.js"></script>
    </head>
    <body>
      <h1 id="qunit-header">Pyempty Hello Test Suite</h1>
      <h2 id="qunit-banner"></h2>
      <div id="qunit-testrunner-toolbar"></div>
      <h2 id="qunit-userAgent"></h2>
      <ol id="qunit-tests"></ol>
      <div id="qunit-fixture">
        <span>lame test markup</span>
        <span>normal test markup</span>
        <span>awesome test markup</span>
      </div>
    </body>
    </html>

自动生成的单元测试也要修改，vi test/hello_test.js

    /*global test, seajs, strictEqual*/

    seajs.use(['../src/hello.js'], function(hello) {
        test('hello test', 1, function() {
            strictEqual(hello.hello('wawa'), 'hello wawa', 'should be "hello wawa"');
        });
    });

搞定，输入grunt，看看可以给我们做什么

    ~/src/github/jsempty/js/sources/hello $ grunt 
    Running "lint:files" (lint) task
    Lint free.

    Running "qunit:files" (qunit) task
    Testing hello.html.OK
    >> 1 assertions passed (18ms)

    Running "concat:dist" (concat) task
    File "dist/hello.js" created.

    Running "min:dist" (min) task
    File "dist/hello.min.js" created.
    Uncompressed size: 243 bytes.
    Compressed size: 175 bytes gzipped (191 bytes minified).

    Done, without errors.

因为我们要在整站使用seaJS,所以我们写的模块也是seaJS能认识的模块，spm和seaJS是配合最好的。
vi package.json,增加output节, spm打包需要。

    {
      "name": "hello",
      "version": "0.1.0",
      "engines": {
        "node": ">= 0.6.0"
      },
      "scripts": {
        "test": "grunt qunit"
      },
      "devDependencies": {
        "grunt": "~0.3.15"
      },
      "output": {
        "hello.js": "[hello.js]"
      }
    }

编译和部署模块，其实可以写成grunt的自定义任务

    cd jsempty/js/source/hello
    spm build
    spm deploy --local=../../ #拷贝到了jsempty/js目录下


## 整体测试

    cd jsempty

vim index.html

    //移除这两行
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
    <script>window.jQuery || document.write('<script src="js/vendor/jquery-1.8.2.min.js"><\/script>')</script>
    //增加如下行
    <script src="js/seajs/1.2.1/sea.js" data-main="./js/main"></script>

vim js/main.js

    seajs.config({
        alias: {
            'jquery': '1.8.1',
            'hello': '0.1.0'
        }
    });

    seajs.use(['hello', 'jquery'], function(Hello, $) {
        $(function(){
            alert(Hello.hello('wawa'));
        }());
    });

启动spm测试服务器

    spm server -p 8002

用浏览器打开http://localhost:8002/index.html

## 小结

好了，第一个javascript程序写好了。
