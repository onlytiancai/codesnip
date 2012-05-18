var wawa = {};

//# 基础组件

//## UrlRouter
wawa.Router = function(){
    function Router(){}

    Router.prototype.setup = function(routemap, defaultFunc){
        var that = this;
        this.routemap = [];
        this.defaultFunc = defaultFunc;
        for (var rule in routemap) {
            if (!routemap.hasOwnProperty(rule)) continue;
            that.routemap.push({
                rule: new RegExp(rule, 'i'),
                func: routemap[rule]
            });             
        }
    };

    Router.prototype.onHashChange= function(){
        console.log(window.location.hash);

        var that = this, hash = location.hash, route, matchResult;
        for (var routeIndex in this.routemap){
            route = this.routemap[routeIndex];
            matchResult = hash.match(route.rule);
            if (matchResult){
                route.func.apply(that.callcontext, matchResult.slice(1));
                return; 
            }
        }
        this.defaultFunc.apply(that.callcontext);

    };

    Router.prototype.start = function(context){
        this.callcontext = context;
        var that = this;
        $(window).bind('hashchange', function() {
            that.onHashChange(); 
        });
        that.onHashChange();
    };

    return Router;
}();

//## ViewManager
wawa.ViewManager = function(){
    function ViewManager(){
        this.models   = {};
        this.views    = {};
        this.actions  = {};
        this.helper   = {};
    }

    ViewManager.prototype.trigger = function(eventname, data){
        jQuery.event.trigger(eventname, data, this);
    };

    ViewManager.prototype.on = function(eventname, handler){     
        $(this).bind(eventname, handler);
    };

    ViewManager.prototype.off = function(){
        $(this).unbind(eventname, handler);
    };

    ViewManager.prototype._addFunc = function(type, name, value){
        var that = this;
        var wrapvalue = function(){value.apply(that, arguments);}; 
        this[type][name] = wrapvalue;
    }

    ViewManager.prototype.addView = function(name, value){this._addFunc('views', name, value)}
    ViewManager.prototype.addAction = function(name, value){this._addFunc('actions', name, value)}
    ViewManager.prototype.addHelper = function(name, value){this._addFunc('helper', name, value)}

    return ViewManager;

}();
