### 关于一个多模块包合并后的问题

这是一个示例项目，直接在这个目录spm server就可以预览了，但使用上我觉得有些不方便。

js 目录下存放各种js模块，其中seajs,jquery是第三方模块，而modules下是项目本身的包。

js/modules/src 目录下有a.js和b.js，以及合并后的test.js, 具体内容看代码。

tests.js打包后大致如下

    define("#test/1.0.0/a", ...);
    define("#test/1.0.0/b", ...);
    define("#test/1.0.0/test", ...);

把它部署在js/test/1.0.0/test.js这里

现在我在页面上使用test这个模块只能下面这样用。
    seajs.config({
        alias: {
            'jquery': '1.8.1',
            'test': '1.0.0'
        } 
    });

    seajs.use(['jquery', 'test'], function($, test){
        $(function(){
            test.b.show();
        }());
    })

其实a和b也打包到test.js这个文件里了,页面上我还想使用a或b模块，比如。

    seajs.use(['jquery', '#test/1.0.0/a'], function($, a){
        $(function(){
            $('body').html(a.hello('wawa'));
        }());
    })

可这段代码是不可能运行的
