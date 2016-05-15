/* == 介绍
 *
 * 流失计算分组信息，可指定分组纬度和指标信息
 *
 * == 测试
 *
 * mocha stream-group.js
 *
 * == todo
 *
 * - 目前只支持一级group，需要支持多级group
 *
 * */

var from = require('from');
var assert = require('assert');

function groupStream(stream, groupby, metric, callback) {
    var ret = {};
    stream.on('data', function(data) {
        console.log('write:', data);
        var group = data[groupby];
        var value = ret[group] || 0;
        ret[group] = value + data[metric];
     }).on('end', function () {
        console.log('end:', ret);
        callback(null, ret);
     });
}

var test_data = [
    {country: 'cn', gender: 'M', people: 100},
    {country: 'cn', gender: 'F', people: 80},
    {country: 'us', gender: 'M', people: 50},
    {country: 'us', gender: 'F', people: 40},
];

describe('group from stream', function() {
    it('group by country', function (done) {
        groupStream(from(test_data), 'country', 'people', 
        function(err, data) {
            assert(data['cn'] == 180);
            assert(data['us'] == 90);
            done();
        })
    });

    it('group by gender', function (done) {
        groupStream(from(test_data), 'gender', 'people', 
        function(err, data) {
            assert(data['M'] == 150);
            assert(data['F'] == 120);
            done();
        })
    });

});
