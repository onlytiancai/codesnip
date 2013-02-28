define(function(require, exports, module) {
    var Backbone = require('backbone');
    var _ = require('underscore');
    var Mustache = require('mustache');

    var tpl = require('./index.html');
    $('.main-body').html(tpl);

    var HelloView = Backbone.View.extend({
        initialize: function(options){
            _.bindAll(this, 'render');
            _.bindAll(this, 'show_hello');
        },
        el: $(".hello_container"),
        template: require('./hello.tpl'),
        events: {
            "click button": "show_hello",
        },
        show_hello: function(){
            alert("hello " + this.model.name);
        },
        render: function(){
            var html = Mustache.render(this.template, this.model);
            $(this.el).html(html);
        }
    });
    var hello_view = new HelloView(); 

    var Workspace = Backbone.Router.extend({
        routes: {
            "": "index", 
            ":name": "index"
        },
        index: function(name){
            hello_view.model = {name: name || "world"};
            hello_view.render();
        }
    });

    var app = new Workspace();
    Backbone.history.start();
});
