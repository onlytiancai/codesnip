define(function(require) {
    var Vue = require('vue');
    Vue.use(require('vue_resource'));

    var vm = new Vue({
        el: '#app',
        data: {
            items: [],
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
    return vm;
});