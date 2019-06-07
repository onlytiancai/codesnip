# require.js + vue + bootstrap

vue 是上手最快的前端 MVVM 框架，require.js 也是经典老牌的前端模块化工具，Bootstrap 是前端最常用的 CSS 框架。

webpack 上手略难，需要安装 nodejs，和很多配置，虽然现今，但暂时不引入。

## require.js 入口

    <script src="https://cdn.staticfile.org/require.js/2.3.6/require.min.js"></script>
    <!-- require.js 配置提取成公共脚本，否则会有很多重复配置-->
    <script src="js/requirejs-config.js"></script>
    <script type="text/javascript">
        // 应用入口，相对 baseUrl 进行解析
        requirejs(['../app/main']);
    </script>

## require.js 配置

    requirejs.config({
        baseUrl: 'js/lib',
        paths: {
            vue: 'https://cdn.staticfile.org/vue/2.4.2/vue.min',
            vue_resource: 'https://cdn.staticfile.org/vue-resource/1.5.1/vue-resource.min',
            jquery: 'https://cdn.staticfile.org/jquery/3.4.1/jquery.min',
            bootstrap: 'https://cdn.staticfile.org/twitter-bootstrap/3.4.1/js/bootstrap.min',
        },
        "shim": {
            "bootstrap": ["jquery"]
        }
    });

## 导入第三方组件和内部组件

    // Load library/vendor modules using
    // full IDs, like:
    var print = require('print');
    print(res.data);
    
    // Load any app-specific modules
    // with a relative require call,
    // like:
    var tooltip = require('./tooltip');
    tooltip();                    

## 使用 vue 和 vue 插件

    var Vue = require('vue');
    // 加载 vue 插件
    Vue.use(require('vue_resource'));

## 使用 jquery 和 bootstrap 组件

    var $ = require('jquery');
    // 加载 bootstrap 组件
    require('bootstrap');
    // 使用 bootstrap 的 tooltip 组件
    $('[data-toggle="tooltip"]').tooltip();
    
## vue 界面更新后使用 bootstrap 组件

    updated: function() {
        $('[data-toggle="tooltip"]').tooltip();
    },
  