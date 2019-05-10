define(function(require) {
    return {
        template: '#foo',
        data: function() {
            // 组件的 data 选项是一个函数，组件不相互影响
            return {
                items: [],
                path: this.$route.path,
            }
        },
        mounted: function() {
            this.get();
        },
        // 需要在 updated 事件里使用 Bootstrap 的 tooltip 组件
        updated: function() {
            // Load any app-specific modules
            // with a relative require call,
            // like:
            var tooltip = require('./tooltip');
            tooltip();
        },
        watch: {
            // 监听路由变化，随时获取新的列表信息
            '$route': 'routeChange'
        },
        methods: {
            routeChange: function() {
                this.path = this.$route.path;
                this.get(this.$route.params.page);
            },
            get: function(page) {
                this.$http.get('data.json?page=' + page).then(function(res) {
                    // Load library/vendor modules using
                    // full IDs, like:
                    var print = require('print');
                    print(res.data);

                    this.items = res.data;
                });
            }
        },
    };
});