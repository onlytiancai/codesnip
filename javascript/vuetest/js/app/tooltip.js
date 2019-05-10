define(function(require) {
    var $ = require('jquery');
    // 加载 bootstrap 组件
    require('bootstrap');
    return function() {
        $('[data-toggle="tooltip"]').tooltip();
    }
});