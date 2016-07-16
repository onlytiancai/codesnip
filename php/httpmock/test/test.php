<?php
require_once(dirname(dirname(__FILE__)) . "/index.php");

$r = new Route(dirname(__FILE__));

echo "==静态文件测试\n";
$r->addRoute("apimock.io", "GET", "/users", "/data/users.json");
$r->dispatch("apimock.io", "GET", "/users");

echo "==动态文件测试\n";
$r->addRoute("apimock.io", "GET", "/users/1", "/get_user.php");
$r->dispatch("apimock.io", "GET", "/users/1");

echo "==远程拉取文件\n";
$r->addRoute("apimock.io", "GET", "/fetchweb", "http://ihuhao.com/");
$r->dispatch("apimock.io", "GET", "/fetchweb");

echo "==文件未找到测试\n";
$r->addRoute("apimock.io", "GET", "/notfound", "/data/notfound");
$r->dispatch("apimock.io", "GET", "/notfound");



