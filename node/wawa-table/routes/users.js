var express = require('express');
var router = express.Router();
var passport = require('passport');

/* GET users listing. */
router.get('/', function(req, res, next) {
  res.send('respond with a resource');
});

router.get('/auth/qq',
  passport.authenticate('qq'),
  function(req, res){
    // The request will be redirected to qq for authentication, so this
    // function will not be called.
});

router.get('/auth/qq/callback', 
  passport.authenticate('qq', { failureRedirect: '/login' }),
  function(req, res) {
    // Successful authentication, redirect home.
    res.redirect('/');
});

module.exports = router;
