var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: '蛙蛙表格', user: req.user ? req.user.nickname : '' });
});

module.exports = router;
