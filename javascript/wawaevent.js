/* module name: wawaevent
 * version: 0.1
 * depend:underscore,backbone
 * description: 处理javascript的异步任务流
 * document:
 *       setup:指定任务流逻辑
 *             1. 可以用sequence来设置一个列表来制定, 如{'sequence': ['worker1', 'worker2', 'worker3']}
 *             1. 可以用逗号隔开多个eventName，然后制定一个handler,如{'worker1.error, worker2.error': 'default_error_handler'}
 *             1. 可以单独指定一个事件的handler,handler除了可以使用任务流名字外
 *                 ，还可以直接使用函数，如{'worker1': function(){console.log('worker1 success')}}
 *       errors:任务流执行过程中的错误信息
 *       handlers: 任务列表
 *       data:所有任务需要的数据和生成的数据都在这里存取，所有任务函数都没有参数
 *       start:启动任务流
 * example:
 *      比如异步任务worker1, worker2, worker3，worker1执行成功再执行worker2，worker2执行成功再执行worker3,
 *      每个异步任务都有自己的错误处理，可以如下这样使用。
 *      em.setup({
 *          'sequence': ['worker1', 'worker2', 'worker3'],
 *          'worker1.error, worker2.error': 'default_error_handler',
 *          'worker1': function(){console.log('worker1 success')}
 *      });
 * */
var wawa = {};
wawa.getEventManager = function(){
    var em = {
        errors: [], 
        handlers: {},
        data: {},
        start: function(){
            this.handlers[this._firstHandlerName]();
        },
        setup: function(options){
            var that = this;
            _.each(this.handlers, function(handler, eventName){
                var context = _.extend({
                    triggerSuccess: function(){that.trigger(eventName)},
                    triggerError: function(){that.trigger(eventName+'.error')}
                }, that);
                that.handlers[eventName] = _.bind(handler, context);
            });
            _.each(options, this._handlerOption)
        },
        _handlerOption: function(value, key){
            var arrEvents = null, that = this;

            //sequence handler
            if (key === 'sequence'){      
                this._bindSequence(key, value);
                return;
            }

            //mutil event names handler
            arrEvents = key.split(', ');
            if (arrEvents.length > 1){
                _.each(arrEvents, function(event_){that._bind(event_, value)});
                return;
            }

            //sigle event name handler
            this._bind(key, value);
        },
        _bindSequence: function(key, value){
            if (value.length < 1) throw {message:'sequence value is empty'};

            var lastEventName = value[0], thisHandlerName = null;
            this._firstHandlerName = value[0];
            for(var i = 1, l = value.length; i < l; i++){
                thisHandlerName = value[i];
                this._bind(lastEventName, thisHandlerName);
                lastEventName = thisHandlerName;
            }
        },
        _bind: function(event_, handler){
            if (typeof handler == 'string'){
                handler = this.handlers[handler]; 
            }
            this.bind(event_, handler);
        },
    };
    em = _.extend(em, Backbone.Events);
    _.bindAll(em, 'start', '_handlerOption');
    return em;
};
/* ==================================================分割线
 * 以下为示例
 * */
var em = wawa.getEventManager();

em.handlers.worker1 = function(){
    if (this.data.name == 'Jim'){
        console.log('worker1:' + this.data.name);
        this.data.name = 'jim green';
        this.triggerSuccess();
    }else{
        this.errors.push('name is not jim');
        this.triggerError();
    }
};

em.handlers.worker2 = function(){
    this.data.op = this.data.op == 'hello' ? 'goodbye': 'hello';
    console.log('worker2:' + this.data.op);
    this.triggerSuccess();
};

em.handlers.worker3 = function(){
    this.data.result =  this.data.op + ' ' + this.data.name;
    console.log('worker3:' + this.data.name);
    this.triggerSuccess();
};

em.handlers.default_error_handler = function(){
    console.log('error:' + this.errors);
};

em.setup({
    'sequence': ['worker1', 'worker2', 'worker3'],
    'worker1.error, worker2.error': 'default_error_handler',
    'worker1': function(){console.log('worker1 success')}
});

em.data.name = 'Jim2';
em.data.op = 'hello';
em.start();
console.log(em.data.result);
