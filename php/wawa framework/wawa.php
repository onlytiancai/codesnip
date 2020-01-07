<?php
ini_set('date.timezone','Asia/Shanghai');
header("Content-type:text/html;charset=utf-8");
error_reporting(E_ALL & ~E_STRICT & ~E_NOTICE);

class Wawa
{    
    private static $_getHandlers = [];
    private static $_postHandlers = [];    
    private static $config = null;
    private static $sitePrefix = null;
    private static $controllerName = null;
    private static $actionName = null;   
        
    
    public function __construct() {
        // 使用 $this 无法访问静态属性，所以要复制一份到 $this 里
        $config = self::$config;        
        $this->config = $config;
        $this->sitePrefix = self::$sitePrefix;
        $this->dsn = "{$config['db']['dbms']}:host={$config['db']['dbhost']};dbname={$config['db']['dbName']};port={$config['db']['dbport']};charset=utf8";        
    }    
    
    public function execute($sql, $args=[]) {
        $conn = $this->_conn();
        $this->_execute($conn, $sql, $args);
        $conn = null;
    }
    
    public function fetch($sql, $args=[]) {      
        $conn = $this->_conn();
        $st = $this->_execute($conn, $sql, $args);        
        $ret = $st->fetchAll();
        $conn = null;
        return $ret;             
    }
    
    public function render($view, $data=[]) {
        $view = $this->_joinpath($view);
        if (!file_exists($view)) die("$view not found."); 
        include($view);
    }    
    
    public function e($str) {
        return htmlspecialchars($str);            
    }
    
    public function isPost() {
        return $_SERVER['REQUEST_METHOD'] == 'POST';
    }
    
    public function redirect($url) {
        if (strpos('http://', $url) !== 0 && strpos('https://', $url) !== 0) $url = $this->sitePrefix . $url;
        header("Location: $url");
    }
    
    public function run() {        
        $this->_doAction();        
    }
    
    public function get($action, $handler) {
        self::$_getHandlers[$action] = $handler;
    }
    
    public function post($action, $handler) {
        self::$_postHandlers[$action] = $handler;
    }
    
    public function send404($msg="File Not Found.\n") {
        header("HTTP/1.0 404 Not Found");
        echo $msg;
        die();
    }
            
    private function _execute($conn, $sql, $args) {
        $st = $conn->prepare($sql);
        $st->execute($args);
        return $st;
    }
    
    private function _conn() {
        try {
            return new PDO($this->dsn, $this->config['db']['dbuser'], $this->config['db']['dbpass']);          
        } catch (PDOException $e) {
            die ("Error!: " . $e->getMessage() . "<br/>");
        } 
    }
           
    private function _doAction() {
        $actionName = self::$actionName;
        $handlers = $this->isPost() ? self::$_postHandlers : self::$_getHandlers;        
        if (empty($handlers[$actionName])) return $this->send404("action $actionName not found.");
        $handlers[$actionName]($this);
    }
    
    /* begin static method */
    public static function init() {
        $config = [];
        $config_file = self::_joinpath('config.php');
        if (!file_exists($config_file)) die("$config_file file not found");
        require_once($config_file);
        
        self::$config = $config;   
        self::$sitePrefix = self::$config['site_prefix'];
    }
    
    public static function runRoute() {
        self::_parseRoute();
        self::_loadController();
    }
    
    private static function _parseRoute() {        
        // 取出 request_uri，去掉 query string 部分: '/me/xxx?id=1' => '/me/xxx'
        $request_uri = $_SERVER['REQUEST_URI'];
        $pos = strpos($request_uri, '?');
        if ($pos !== false) $request_uri = substr($request_uri, 0, $pos);   
        
        
        // 去掉 sitePrefix 前缀：'/me/xxx' => 'xxx'
        $pos = strpos($request_uri, self::$sitePrefix);
        if ($pos === 0) $request_uri = substr($request_uri, strlen(self::$sitePrefix));  

        // 取出 controllerName 和 actionName
        $arr = explode('/', $request_uri);
        self::$controllerName =  empty($arr[0]) ? 'index' : $arr[0];
        self::$actionName = empty($arr[1]) ? 'index' : $arr[1];
    }
    
    private static function _loadController() {
        $controllerFile = self::_joinpath('controllers/' . self::$controllerName . '.php');
        if (!file_exists($controllerFile)) return self::send404("controller {$this->controllerName} not found.");        
        include($controllerFile);
    }   
    
    private static function _joinpath($path) {        
        return join(DIRECTORY_SEPARATOR, [__DIR__, $path]);    
    }
    /* end static method */
}