<?php
$file = './people.txt';

header("Access-Control-Allow-Origin: *");
header('Access-Control-Allow-Methods: PUT, GET, POST, DELETE, OPTIONS');
header("Access-Control-Allow-Headers: X-Requested-With");
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] == 'POST' ) {
    $list = json_decode(file_get_contents("php://input"));
    foreach ($list as $data) {
        file_put_contents($file, implode(',', $data)."\n", FILE_APPEND | LOCK_EX);
    }
    echo "save success";
}
