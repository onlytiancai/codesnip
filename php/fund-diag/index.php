<?php
error_reporting(E_ALL);
use Wruczek\PhpFileCache\PhpFileCache;
require_once __DIR__ . "/vendor/autoload.php";

$cache = new PhpFileCache();

$data = $cache->refreshIfExpired("simple-cache-test", function () {
    return date("H:i:s"); // return data to be cached
}, 10);

echo "Latest cache save: $data";
