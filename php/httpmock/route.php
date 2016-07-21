<?php

class Route
{
    private $routes = array();
    private $basedir = null; 

    function __construct($basedir="") {
        $this->basedir = $basedir ? $basedir : dirname(__FILE__);
    }

    public function addRoute($host, $method, $url, $action)
    {
        array_push($this->routes, array($host, $method, $url, $action));
    }

    public function dispatch($host, $method, $url)
    {
        $found = array_filter($this->routes, function ($x) use ($host, $method, $url){
            return $x[0] == $host && $x[1] == $method && $x[2] == $url;
        });

        if (empty($found)) return $this->_notfound();
        $found = reset($found);
        $this->_dispatch($found[3]);
    }

    private function _notfound()
    {
        echo "file not found. \n";
    }

    private function _dispatch($action)
    {
        $file = $this->basedir . $action;
        if (file_exists($file)) {
            $ext = pathinfo($file, PATHINFO_EXTENSION);
            
            // 静态文件处理
            if ($ext != "php") return fpassthru(fopen($file, 'r'));

            // PHP动态文件处理
            return require($file);
        } 

        // 远程抓取文件，透明代理 
        if (strpos($action, "http:") === 0 || strpos($action, "https:")) {
            echo file_get_contents($action);
            return;
        }
        
        // 未找到处理文件
        return $this->_notfound(); 
    }
}
