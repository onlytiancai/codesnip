var debug = require('debug')('wawa');
var mysql = require('mysql');
var config = require('config');

var pool = mysql.createPool(config.get('db'));
debug(config.get('db'));

// 用户类型
exports.USERTYPE_QQ = 0;
exports.USERTYPE_WEIXIN = 1;


/**
 * 注册用户
 * */
exports.registerUser = function (usertype, openid, username, ip, callback) {
  var sql = "insert into users (usertype, openid, username, status, registerTime, "
          + "registerIp, lastLoginTime, lastLoginIp) "
          + "values (?, ?, ?, ?, now(), ?, ?, ?)";

  pool.query(sql,
    [usertype, openid, username, 0, ip, new Date(), ip],
    function(err, results) {
      callback(err);
    }
  );
};

/**
 * 判断用户名是否存在
 * */
exports.existsUsername = function (username, callback) {
  pool.query(
    'select * from users where username = ? limit 1',
    [username],
    function(err, rows, fields) {
      callback(null, rows.length !== 0);
    }
  );
}

/**
 * 判断openid是否存在
 * */
exports.existsOpenid = function (usertype, openid, callback) {
  pool.query(
    'select * from users where usertype = ? and openid = ? limit 1',
    [usertype, openid],
    function(err, rows, fields) {
      callback(null, rows.length !== 0);
    }
  );
}

/**
 * 记录用户登录
 * */
exports.logLogin = function (userid, ip) {

}

/**
 * 获取用户
 * */
exports.getUser = function (usertype, openid, callback) {
  exports.existsOpenid(usertype, openid, function(err, existing) {
    if (existing) return callback('用户不存在');
    pool.query(
      'select * from users where usertype = ? and openid = ? limit 1',
      [usertype, openid],
      function(err, rows, fields) {
        callback(null, rows[0]);
      }
    );
  });
}

exports.removeUser = function (usertype, openid, callback) {
  pool.query(
    'delete from users where usertype = ? and openid = ? limit 1',
    [usertype, openid],
    function(err, result) {
      callback(null, result.affectedRows);
    }
  );
}
