define(function(require, exports, module) {
    var $ = require('jquery'),
        hello = require('./a'),
        name = 'seajs';

    exports.show = function(){
        $('body').html(hello.hello(name));
    };
});
