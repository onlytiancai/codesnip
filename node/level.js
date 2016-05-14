var levelup = require('levelup');
var memdown = require('memdown');
var async = require('async');
var assert = require('chai').assert;

var db = levelup({ db: memdown });

db.put('name', 'LevelUP', function (err) {
  if (err) return console.log('Ooops!', err)

  db.get('name', function (err, value) {
    if (err) return console.log('Ooops!', err)

    console.log('name=' + value)
  })
})

function addEvent(appid, time, data, callback) {
    console.log('addEvent', arguments);
    callback(null);
}

function queryEvents(appid, event_name, callback) {
    var ret = [
        {time: new Date(2016, 4, 14, 17, 17, 0), value: 9}, 
        {time: new Date(2016, 4, 14, 17, 18, 0), value: 2}, 
    ];
    callback(null, ret);
}


var test_data = [
    {time: new Date(2016, 4, 14, 17, 17, 0), data: {event_name: 'click', value: 3}},
    {time: new Date(2016, 4, 14, 17, 17, 3), data: {event_name: 'click', value: 2}},
    {time: new Date(2016, 4, 14, 17, 17, 5), data: {event_name: 'click', value: 4}},
    {time: new Date(2016, 4, 14, 17, 18, 0), data: {event_name: 'click', value: 1}},
    {time: new Date(2016, 4, 14, 17, 18, 2), data: {event_name: 'click', value: 1}},
];

var test_appid = 10000;

describe('leveldb', function() {
  describe('#hook-and-map-reduce', function () {
    it('will ok', function (done) {

        async.map(test_data, function(data, callback){

            addEvent(test_appid, data.time, data.data, callback)

        },function(){

            queryEvents(test_appid, 'click', function(err, data){
                console.log(data);
                assert.equal(2, data.length);

                assert.equal(new Date(2016, 4, 14, 17, 17, 0).getTime(), data[0].time.getTime());
                assert.equal(9, data[0].value);

                assert.equal(new Date(2016, 4, 14, 17, 18, 0).getTime(), data[1].time.getTime());
                assert.equal(2, data[1].value);

                done(); 
            }); 

        });
    });
  });
});
