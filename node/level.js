/*
== 介绍：

使用leveldb来接受事件流，并进行实时计算。
目前是只做根据event_name纬度进行1分钟力度的加总。

== 测试： 

    mocha level.js

== todo:

- moments进行时间多次转换，有些浪费
- 统计纬度，粒度，聚合函数可配置化

*/
var levelup = require('levelup');
var memdown = require('memdown');
var moment = require('moment');
var async = require('async');
var assert = require('assert');
var uuid = require('node-uuid');

var db = levelup({ db: memdown });
//var db = levelup('./mydb');

function _statEvent(appid, time, data, callback) {
    var time = moment(time).format('YYYYMMDDHHmm');
    var key = ['stat-events', appid, data.event_name, time].join(':');

    db.get(key, function (err, old_value) {
        old_value = err ? {sum: 0} : JSON.parse(old_value); 
        var value = {sum: old_value.sum + data.value};

        console.log('statEvent:', key, value, old_value, data.value);
        db.put(key, JSON.stringify(value), callback);
    })
}

var q = async.queue(function (task, callback) {
    var event_id = uuid.v4();
    var event_time = moment(task.time).format('YYYYMMDDHHmmss');
    var key = ['events', task.appid, event_time , event_id].join(':'); 

    _statEvent(task.appid, task.time, task.data, function(err){
        if (err) return console.log('statEvent err')
        db.put(key, JSON.stringify(task.data), callback);
    });

}, 1);


function addEvent(appid, time, data, callback) {
    console.log('addEvent:', appid, time, data);
    q.push({appid: appid, time: time, data: data}, callback);
}


function queryEvents(appid, event_name, callback) {
    var ret = [];
    var start = ['stat-events', appid, event_name, ''].join(':');
    var end = ['stat-events', appid, event_name, '~'].join(':');

    db.createReadStream({ start: start, end: end })
    .on('data', function (data) {
        console.log('stream-data:', data.key, '=', data.value)

        var time = moment(data.key.split(':')[3], 'YYYYMMDDHHmm').toDate();
        var value = JSON.parse(data.value);
        ret.push({time: time, sum: value.sum});
    })
    .on('end', function () {
        callback(null, ret);
    });

}

// ============= unit test

var test_data = [
    {time: new Date(2016, 4, 14, 17, 17, 0), data: {event_name: 'click', value: 3}},
    {time: new Date(2016, 4, 14, 17, 17, 3), data: {event_name: 'click', value: 2}},
    {time: new Date(2016, 4, 14, 17, 17, 5), data: {event_name: 'click', value: 4}},
    {time: new Date(2016, 4, 14, 17, 18, 0), data: {event_name: 'click', value: 1}},
    {time: new Date(2016, 4, 14, 17, 18, 2), data: {event_name: 'click', value: 1}},
];

var test_appid = 10000;

describe('leveldb', function() {
  describe('#stat-events', function () {
    it('will ok', function (done) {

        async.map(test_data, function(data, callback){
            addEvent(test_appid, data.time, data.data, callback)
        },function(){

            queryEvents(test_appid, 'click', function(err, data){
                console.log(data);
                assert(2 == data.length);

                assert(new Date(2016, 4, 14, 17, 17, 0).getTime() == data[0].time.getTime());
                assert(9 == data[0].sum);

                assert(new Date(2016, 4, 14, 17, 18, 0).getTime() == data[1].time.getTime());
                assert(2 == data[1].sum);

                done(); 
            }); 

        });
    });
  });
});
