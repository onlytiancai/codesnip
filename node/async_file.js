//http://howtonode.org/control-flow-part-iii
var fs = require('fs');

// Here is the async version without helpers
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
    console.log("loaddir:", x.length);
  });
});


var async = require('async');

// 用async重构下
function loaddir2(path, callback) {
    var all_files = [], 
        real_files = [];

    async.series([
        function(callback){
            fs.readdir(path, function (err, filenames) {
                all_files = filenames;
                callback(err);
            });
        },
        function(callback){
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
        function(callback){
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
        console.log("loaddir2", x.length);
    });
});
