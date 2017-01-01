var Promise = require("bluebird");
var user_model = require('../models/user_model');


var usertype = user_model.USERTYPE_QQ,
  openid = '1',
  username = 'testuser',
  ip = '0.0.0.0';

describe('user_model', function () {
  describe('registerUser', function () {

    it('node callback should success', function (done) {
      user_model.removeUser(usertype, openid, function () {
        registerUser(done);
      });
    });

    it('promise should success', function (done) {
      user_model.removeUser(usertype, openid, function () {
        registerUser(done);
      });
    });

    it('promise2 should success', function (done) {
      user_model.removeUser(usertype, openid, function () {
        registerUser3().then(done).catch(done);
      });
    });


  });
});

function registerUser(callback) {
  user_model.existsOpenid(usertype, openid, function (err, existing) {
    if (err) return callback(err);
    if (existing) return callback('您已注册');
    user_model.existsUsername(username, function (err, existing) {
      if (err) return callback(err);
      if (existing) return callback('用户名已存在');
      user_model.registerUser(usertype, openid, username, ip, function (err) {
        callback(err);
      });
    });
  });
}

userModelPromise = Promise.promisifyAll(user_model);

function registerUser2(callback) {
  userModelPromise.existsOpenidAsync(usertype, openid)
    .then(function(exists) {
      if (exists) return callback('您已注册');
      userModelPromise.existsUsernameAsync(username)
        .then(function(exists) {
          if (exists) return callback('用户名已存在');
            userModelPromise.registerUserAsync(usertype, openid, username, ip)
              .then(function() {
                callback();
              })
              .catch(function (err) {
                callback(err);
              });
        })
        .catch(function(err) {
          callback(err);
        });
    })
    .catch(function (err) {
      callback(err);
    });
}

function registerUser3() {
  return new Promise(function(resolve, reject) {
    userModelPromise.existsOpenidAsync(usertype, openid).then(function(exists) {
      if (exists) return reject('您已注册'); 
      return userModelPromise.existsUsernameAsync(username);
    }).then(function (exists) {
      if (exists) return reject('用户名已存在'); 
      return userModelPromise.registerUserAsync(usertype, openid, username, ip);
    }).then(function() {
      resolve();
    })
    .catch(function(err) {
      reject(err);
    });
  }); 
}


