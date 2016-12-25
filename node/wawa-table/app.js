var express = require('express');
var path = require('path');
var favicon = require('serve-favicon');
var weblogger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var passport = require('passport');
var QQStrategy = require('passport-qq').Strategy;
var config = require('config');
var session = require('express-session');
var MySQLStore = require('express-mysql-session')(session);
var logger = require('./libraries/logger').logger;

logger.info('app initing');


var index = require('./routes/index');
var users = require('./routes/users');

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

// uncomment after placing your favicon in /public
//app.use(favicon(path.join(__dirname, 'public', 'favicon.ico')));
app.use(weblogger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());

var options = config.get('db');
options['createDatabaseTable'] = false;
app.use(session({
    key: config.get('session.key'),
    secret: config.get('session.secret'),
    store: new MySQLStore(config.get('db')),
    resave: true,
    saveUninitialized: true,
}));


app.use(passport.initialize());
app.use(passport.session());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', index);
app.use('/users', users);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

passport.use(new QQStrategy({
    clientID: config.get('oauth-qq.appid'),
    clientSecret: config.get('oauth-qq.appkey'),
    callbackURL: config.get('oauth-qq.callback'), 
  },
  function(accessToken, refreshToken, profile, done) {
    done(null, profile); }
));

passport.serializeUser(function(user, done) {
    done(null, user);
});

passport.deserializeUser(function(user, done) {
    done(null, user);
});

module.exports = app;
