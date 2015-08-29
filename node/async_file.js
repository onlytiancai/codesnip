// 读取某录下的所有文件内容放到一个结果数组里
// http://howtonode.org/control-flow-part-iii
var fs = require('fs');

// 原始版本 
function loaddir(path, callback) {
  fs.readdir(path, function (err, filenames) {
    if (err) { callback(err); return; }
    var realfiles = [];
    var count = filenames.length;
    filenames.forEach(function (filename) {
      fs.stat(filename, function (err, stat) {
        if (err) { callback(err); return; }
        if (stat.isFile()) {
          realfiles.push(filename);
        }
        count--;
        if (count === 0) {
          var results = [];
          realfiles.forEach(function (filename) {
            fs.readFile(filename, function (err, data) {
              if (err) { callback(err); return; }
              results.push(data);
              if (results.length === realfiles.length) {
                callback(null, results);
              };
            });
          });
        }
      });
    });
  });
}


// And it's used like this
loaddir(__dirname, function (err, result) {
  if (err) throw err;
  result.forEach(function(x){
    console.log("loaddir1:", x.length);
  });
});


var async = require('async');

// 用async重构下
function loaddir2(path, callback) {
    var all_files = [], 
        real_files = [];

    async.series([
        function(callback){ // 1. 读出所有子目录和文件名
            fs.readdir(path, function (err, filenames) {
                all_files = filenames;
                callback(err);
            });
        },
        function(callback){ // 2. 过滤掉子目录，剩下文件
            var filter_err = null;
            async.filter(all_files, function(file, callback2) {
                fs.stat(file, function (err, stat) {
                    filter_err = err; 
                    callback2(stat.isFile());
                });
            }, function(results){
                real_files = results;
                callback(filter_err);
            });
        },
        function(callback){ // 3. 把文件内容读出来，放结果数组里
            async.map(real_files, function(file, callback2){
                fs.readFile(file, function (err, data) {
                    callback2(err, data);
                });
            }, function(err, results){
                callback(err, results); 
            }); 
        }
    ], function(err, results){
        callback(err, results.pop()); 
    });
}


loaddir2(__dirname, function (err, result) {
    result.forEach(function(x){
        console.log("loaddir2:", x.length);
    });
});

function loaddir3(path, callback) {
    var concat_files = function(files, callback){
        async.map(files, function(file, callback){
            fs.stat(file, function (err, stat) {
                if (err) { return callback(err); }
                if (stat.isFile()){
                    fs.readFile(file, function (err, data) {
                        callback(err, data);
                    });
                } //TODO: else 这里不回调可以吗？
            });

        }, callback);
    };

    var compose = async.compose(concat_files, fs.readdir);
    compose(path, callback);
}


loaddir3(__dirname, function (err, result) {
    result.forEach(function(x){
        console.log("loaddir3:", x.length);
    });
});

function loaddir4(path, callback) {
    var concat_files = function(files, callback){
        async.map(files, function(file, callback){
            fs.readFile(file, function (err, data) {
                callback(err, data);
            });

        }, callback);
    };

    var filter_files = function(files, callback){
        var filter_err = null; 
        async.filter(files, function(file, callback2) {
            fs.stat(file, function (err, stat) {
                filter_err = err; 
                callback2(stat.isFile());
            });
        }, function(results){
            real_files = results;
            callback(filter_err, results);
        });

    };

    var compose = async.compose(concat_files, filter_files, fs.readdir);
    compose(path, callback);
}

loaddir4(__dirname, function (err, result) {
    result.forEach(function(x){
        console.log("loaddir4:", x.length);
    });
});

