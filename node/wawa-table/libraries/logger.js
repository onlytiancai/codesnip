var config = require('config');
var path = require('path');
var winston = require('winston');
require('winston-daily-rotate-file');

winston.handleExceptions(
  new winston.transports.File({
    filename: path.join(config.get('logger.path'), config.get('appname') + "-exception"),
  })
);

var transport = new winston.transports.DailyRotateFile({
  filename: path.join(config.get('logger.path'), config.get('appname')),
  datePattern: '-yyyyMMdd.log',
  level: config.get('logger.level'),
});

var options = {
  transports: [transport],
  exitOnError: false,
};

if (config.get('logger.console')) {
  options.transports.push(new (winston.transports.Console)());
}

logger = new (winston.Logger)(options);
exports.logger = logger; 
