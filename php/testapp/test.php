<?php
require 'vendor/autoload.php';

use Noodlehaus\Config;

$conf = Config::load('config.ini');
$logfile = $conf->get('app.logfile', 'app.log');

echo sprintf("logfile = %s\n", $logfile);

$log = new Monolog\Logger('name');
$log->pushHandler(new Monolog\Handler\StreamHandler($logfile, Monolog\Logger::WARNING));

$log->addWarning('Foo');
