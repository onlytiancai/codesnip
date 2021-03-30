<?php
ini_set("display_errors", "On");
ini_set("error_reporting",E_ALL);

require_once __DIR__ . "/vendor/autoload.php";

class Action 
{
    private $api_url = 'https://api.doctorxiong.club';
    const CACHE_TTL = 30*24*60*60;
    private $allow_actions = ['searchFund', 'getPosition'];

    function __construct()
    {
        $this->cache = new Wruczek\PhpFileCache\PhpFileCache();
        $this->http_client = new \GuzzleHttp\Client();
    }

    function searchFund()
    {
        $all_funds = $this->getAllFund();
        $keyword = isset($_GET['keyword']) ? $_GET['keyword'] : '';
        if (mb_strlen($keyword) < 2) return $this->showError('keyword error');

        $result = array_slice(array_filter($all_funds, function($x) use ($keyword){
            return strpos($x[2], $keyword) !== false;
        }), 0, 10, false);

        $result = array_map(function($x) {
            return ['name' => $x[2], 'code'=> $x[0]];
        }, $result);

        $this->responseJson($result);
    }

    function getPosition()
    {
        $code = isset($_GET['code']) ? $_GET['code'] : '';
        if (strlen($code) < 5) return $this->showError('code error');
        $result = $this->getFundPosition($code);
        $this->responseJson($result);
    }

    function getAllFund()
    {
        $data = $this->cache->refreshIfExpired("all-fund", function () {
            $response = $this->http_client->get("{$this->api_url}/v1/fund/all", ['timeout' => 20]);
            return (string)$response->getBody();
        }, self::CACHE_TTL);   
        $data = json_decode($data, true);
        return $data['data'];
    }

    function getFundPosition($code)
    {
        $data = $this->cache->refreshIfExpired("position-$code", function () use ($code){
            $response = $this->http_client->get("{$this->api_url}/v1/fund/position?code=$code");
            return (string)$response->getBody();
        }, self::CACHE_TTL);
        $data = json_decode($data, true);
        return $data['data'];
    }

    function run()
    {
        $action = isset($_GET['action']) ? $_GET['action'] : '';
        if (!in_array($action, $this->allow_actions)) return $this->responseCode(404);
        $this->{$action}();
    } 

    function responseCode($code)
    {
        http_response_code($code);
        exit();
    }

    function responseJson($result)
    {
        header('Content-Type: text/javascript; charset=UTF-8');
        echo json_encode($result, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
        exit();
    }

    function showError($message)
    {
        exit($message);    
    }
}

$action = new Action();
$action->run();
