define(function(require) {
    var Vue = require('vue');
    // 加载 vue 插件
    Vue.use(require('vue_resource'));

    var vm = new Vue({
        el: '#app',
        data: {
            items: [],
        },
        // 需要在 updated 事件里使用 Bootstrap 的 tooltip 组件
        updated: function() {
            // Load any app-specific modules
            // with a relative require call,
            // like:
            var tooltip = require('./tooltip');
            tooltip();
        },
        methods: {
            get: function() {
                this.$http.get('data.json').then(function(res) {
                    // Load library/vendor modules using
                    // full IDs, like:
                    var print = require('print');
                    print(res.data);

                    this.items = res.data;
                });
            }
        }
    });

    var $ = require('jquery');
    $(document).ready(function() {
        vm.get();
    });
});