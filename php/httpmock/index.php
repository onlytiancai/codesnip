<?php
include "route.php";
$r = new Route(dirname(__FILE__));

// begin define route

$r->addRoute("www.dnspod.cn", "GET", "/Login", "/proxy.php");

// end define route

$r->dispatch($_SERVER['SERVER_NAME'], $_SERVER['REQUEST_METHOD'], $_SERVER['SCRIPT_NAME']);
