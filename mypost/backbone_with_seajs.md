### 静态文件目录整体规划
网站的static目录下有如下子目录
```
.
├── css         
├── img
├── js
├── myjs
└── sea-modules
```

1. css目录存放整体性的css，比如bootstrap,网站主layout的css等,
    一些插件的css和插件本身的js文件放在同目录就行了，不用单独放到这里，不能太死板。
1. img目录下放图片，包括bootstrap使用的图片，网站本身使用的图片等。
1. js目录存放第三方的javascript脚本，如jquery和jquery的插件等，这些脚本不符合seajs
    规范，但也可以用seajs去require，而且像jquery这样最最常用的脚本，我就直接在html里引入
    了。
    1. 就把jquery当作浏览器自带的功能，而假设jquery内置的话，jquery
    插件就可以直接用seajs区require了，省得下载一个jquery插件还得去包装成seajs格式的。
    1. 再有就是像bootstrap的一些脚本是依赖jquery的，而且整个站都用，没必要为了实现洁癖把它们都用
    seajs异步加载。
1. myjs里存放我们自己写的代码，除了这个目录里的代码，其它目录的代码都是网上下载的，有可能会在升级时覆盖。
1. sea-modules目录存放seajs格式的第三方javascript包,具体里面的目录安排下面说。


### sea-modules目录

sea-modules目录下的包都用spm install安装，首先是按包的名字建立各个子目录，每个包的目录里
每个版本再建立一个子目录，版本目录里存放具体的源文件，包括开发调试版本和生产环境的压缩版本。
如果要更新某个包的话也可以直接用spm来更新，然后sejs.config里可以随时切换版本，下面会看到。

### myjs目录
这个目录下存放我们编写的js代码，要求全部符合seajs规范。
首先下面建立一个utils和stuff目录，分别存放每个单页面app所要用到的helper包和杂项包。
然后其它的目录基本上是一个子目录表示一个单页面app，单页面app用Backbone开发，每个app目录里面做如下约定。

1. main.js 用来引导整个app，包含了seajs.config和seajs.use的调用，具体下面会看到。
1. app.js 用来做Backbone的初始化工作，包含Backbone的Router的定义，以及驱动路由的Backbone.History.start()语句。
1. *-view.js 用来实现app的每个View。
1. *.tpl 用来存放view使用的模板文件，使用seajs按需异步加载。
1. *.css 用来存放app UI所需要的样式文件，使用seajs按需异步加载。

### 依赖关系

每个app基本上是一个独立的个体，实现高内聚，尽量少的外部依赖。除了本目录内的view,model,
tpl的依赖外，再就是第三方组件(Backbone,Mustache等)的依赖了和项目公共组件的依赖了如require('../utils/xxx.js')。

也就是说如果有多个app使用的共用代码，也保存在myjs目录下，并建立子目录进行命名空间划分，如stuff,utils,xxxhelper等。
这样下来一个网站有很多个app，整个文件目录也很有条理，不会乱。

### main.js 开发

每个app里的这个文件做一些seajs的配置和app的引导，大致如下。

```
seajs.config({                                                                 
    alias: {                                                                   
        "underscore": "underscore/1.4.4/underscore-debug",                     
        "backbone": "backbone/0.9.2/backbone-debug",                           
        "mustache": "mustache/0.5.0/mustache-debug",                           
        "cookie": "cookie/1.0.2/cookie-debug",                                 
        "moment": "moment/1.7.2/moment-debug"                                  
    },                                                                         
    preload: ['seajs/plugin-text'],                                            
    map: [                                                                     
        ['.tpl', '.tpl?201304261457'],                                                    
        ['.js', '.js?201304261457']
    ]                                                                          
});                                                                            
                                                                               
seajs.use(['/static/myjs/add-domain/app'], function(main){                     
    $(function(){                                                              
        main.init();  
    });                                                                        
});
```
1. 首先alias里声明第三方的依赖，这里以后要想升级组件版本，或者使用调试版本的话就修改这里。
1. preload里加载必要的seajs插件，最常用的就是用来加载模板和css的插件了。
1. map里做一个各种文件的映射，如果app升级后记着修改这里的版本号用来清空缓存。
1. seajs.use里直接使用app目录下的app.js文件，并约定每个app.js都export一个init方法,
调用init最好在document.ready的回调里，因为这时候layout的dom都加载好了，app里可能会用到。
而且jquery是在html里载入的，这里可以放心的使用$(function(){})。

### 关于打包

原则上是可以把每个app打包成一个js文件，以减少http请求次数。
但以前折腾过好久，在github上发了很多个issue，也没理解为啥打出来的包各种不能用。
我在app目录下建立了一个package.json
```
{
    "name":"app",
    "output":{"app.js":"."}
}
```
执行spm build --src . 竟然提示不能加载../utils/utils.js，我手工把app.js里的../修改成../../再执行，能打包成功了。
然后我把main.js里修改成seajs.use(['/static/myjs/domains/dist/app-debug']),结果打包后的模块导出的不是app.js，而是utils.js，根本没有init方法。
总之spm打包各种问题，所以再次放弃了，就调试模式跑吧。

### 关于测试

不会。

### 目录安排示例

```
.
├── css
│   ├── bootstrap.css
│   ├── bootstrap.min.css
│   ├── bootstrap-responsive.css
│   ├── bootstrap-responsive.min.css
│   └── main.css
├── img
│   ├── global_sprite.png
│   ├── glyphicons-halflings.png
│   ├── logo.png
│   └── simple.png
├── js
│   ├── bootstrap.js
│   ├── bootstrap.min.js
│   ├── china-zh.js
│   ├── highcharts.src.js
│   ├── jquery-1.8.2.min.js
│   ├── jquery.vector-map.css
│   ├── jquery.vector-map.js
│   ├── jquery.zclip.js
│   └── ZeroClipboard.swf
├── myjs
│   ├── add-domain
│   │   ├── app.js
│   │   ├── domains-view.js
│   │   ├── domains-view.tpl
│   │   ├── main.js
│   │   ├── records-view.js
│   │   └── records-view.tpl
│   ├── charthelpers
│   │   ├── chart.js
│   │   ├── chinamap.js
│   │   ├── isp_chart.js
│   │   ├── region_chart.js
│   │   └── time_chart.js
│   ├── day_report
│   │   ├── app.js
│   │   ├── main.js
│   │   ├── template.tpl
│   │   └── view.js
│   ├── domains
│   │   ├── app.js
│   │   ├── domains.tpl
│   │   └── main.js
│   ├── stuff
│   │   └── moment-zh-cn.js
│   └── utils
│       ├── data_map.js
│       └── utils.js
└── sea-modules
    ├── backbone
    │   └── 0.9.2
    │       ├── backbone-debug.js
    │       ├── backbone.js
    │       └── package.json
    ├── cookie
    │   └── 1.0.2
    │       ├── cookie-debug.js
    │       ├── cookie.js
    │       ├── package.json
    │       └── src
    │           └── cookie.js
    ├── jquery
    │   └── 1.8.3
    │       ├── gallery
    │       ├── jquery-debug.js
    │       ├── jquery.js
    │       └── package.json
    ├── marked
    │   └── 0.2.4
    │       ├── marked-debug.js
    │       ├── marked.js
    │       └── package.json
    ├── moment
    │   └── 1.7.2
    │       ├── moment-debug.js
    │       ├── moment.js
    │       └── package.json
    ├── mustache
    │   └── 0.5.0
    │       ├── mustache-debug.js
    │       ├── mustache.js
    │       └── package.json
    ├── seajs
    │   └── 1.3.0
    │       ├── package.json
    │       ├── plugin-base.js
    │       ├── plugin-text.js
    │       ├── sea-debug.js
    │       └── sea.js
    └── underscore
        ├── 1.4.2
        │   ├── package.json
        │   ├── underscore-debug.js
        │   └── underscore.js
        └── 1.4.4
            ├── underscore-debug.js
            └── underscore.js
```
