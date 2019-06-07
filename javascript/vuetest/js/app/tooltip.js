define(function(require) {
    var $ = require('jquery');
    // 加载 bootstrap 组件
    require('Bootstrap');
    return function() {
        $('[data-toggle="tooltip"]').tooltip();
    }
});