define("#test/1.0.0/a-debug", [], function(require, exports, module) {
    exports.hello = function(name){
        return 'hello ' + name;
    };
});


define("#test/1.0.0/b-debug", ["./a-debug", "#jquery/1.8.1/jquery-debug"], function(require, exports, module) {
    var $ = require('#jquery/1.8.1/jquery-debug'),
        hello = require('./a-debug'),
        name = 'seajs';

    exports.show = function(){
        $('body').html(hello.hello(name));
    };
});


define("#test/1.0.0/test-debug", ["./a-debug", "./b-debug", "#jquery/1.8.1/jquery-debug"], function(require, exports, module) {
    var a = require('./a-debug');
        b = require('./b-debug');

    module.exports = {
        a: a,
        b: b
    };
});
