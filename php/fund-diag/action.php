<?php
ini_set("display_errors", "On");
ini_set("error_reporting",E_ALL);

require_once __DIR__ . "/vendor/autoload.php";

class Action 
{
    private $api_url = 'https://api.doctorxiong.club';
    private $allow_actions = ['searchFund'];

    function __construct()
    {
        $this->cache = new Wruczek\PhpFileCache\PhpFileCache();
        $this->http_client = new \GuzzleHttp\Client();
    }

    function searchFund()
    {
        $all_funds = json_decode($this->getAllFund(), true);
        print_r($all_funds['data'][1]);
    }

    function getAllFund()
    {
        return $this->cache->refreshIfExpired("all-fund", function () {
            $response = $this->http_client->get("{$this->api_url}/v1/fund/all");
            return (string)$response->getBody();
        }, 4*60*60);   
    }

    function run()
    {
        $action = isset($_GET['action']) ? $_GET['action'] : '';
        if (in_array($action, $this->allow_actions)) {
            $this->{$action}();
        } else {
            http_response_code(404);
        }
        exit();
    } 
}

$action = new Action();
$action->run();
