var user_model = require('../models/user_model');

var usertype = user_model.USERTYPE_QQ,
  openid = '1',
  username = 'testuser',
  ip = '0.0.0.0';

describe('user_model', function () {
  describe('registerUser', function () {
    it('should success', function (done) {
      user_model.removeUser(usertype, openid, function () {
        registerUser(done);
      });
    });
  });
});

function registerUser(callback) {
  user_model.existsOpenid(usertype, openid, function (err, existing) {
    if (existing) return callback('您已注册');
    user_model.existsUsername(username, function (err, existing) {
      if (existing) return callback('用户名已存在');
      user_model.registerUser(usertype, openid, username, ip, function (err) {
        callback(err);
      });
    });
  });
}
