define(function(require) {
    var Vue = require('Vue');

    // 加载 vue 插件
    Vue.use(require('VueResource'));
    var VueRoute = require('VueRoute');
    Vue.use(VueRoute);

    var Foo = require('./foo');

    var Page404 = {
        template: '<div>404</div>'
    };

    new Vue({
        router: new VueRoute({
            routes: [{
                path: '/foo/:page',
                component: Foo
            }, {
                path: '*',
                component: Page404
            }]
        }),
    }).$mount('#app');
});