<?php
include "route.php";
$r = new Route(dirname(__FILE__));

// begin define route

$r->addRoute("www.dnspod.cn", "GET", "/Login", "/proxy.php");
$r->addRoute("www.dnspod.cn", "GET", "/yantai/ie6.html", "/proxy.php");
$r->addProxyHost("www.dnspod.cn");

// end define route

$r->dispatch($_SERVER['SERVER_NAME'], $_SERVER['REQUEST_METHOD'], $_SERVER['DOCUMENT_URI']);
