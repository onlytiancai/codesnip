<?php
// https://github.com/qdsang/php-http-proxy

date_default_timezone_set('PRC');

$chunked = 0;
$gziped = 0;
function header_function($ch, $header){
    global $debug;
    if (stripos($header,'chunked') !== false) {
        //如果碰到分断好像会出错,所以过滤一下.若你有好的方案烦告之.
        //关闭掉了。大坑
        //$GLOBALS['chunked'] = 1;
    }
    if (stripos($header,'Content-Encoding: gzip') === 0) {
        $GLOBALS['gziped'] = 1;
    }
    header($header);
    empty($debug) || fwrite($debug,$header);
    return strlen($header);
}

function write_function($ch, $body){
    global $debug;
    if ($GLOBALS['chunked']){
        printf("%x\r\n%s\r\n", strlen($body), $body);
    }else{
        echo $body;
    }
    if ($debug)
    {
        //if ($GLOBALS['gziped'])
        //    fwrite($debug,gzinflate(substr($body,10,-4)));
        //else
            fwrite($debug,$body);
    }
    return strlen($body);
}

function proxy()
{
    global $debug;
    $hearer = array();
    //获取HTTP相关的HEADER信息
    if (function_exists('getallheaders'))
    {
        $allheader = getallheaders();
        foreach($allheader as $h=>$key)
        {
            $header[] = $h.': '.$key;
        }
    }
    else
    {
        foreach($_SERVER as $key=>$value)
        {
            if (strcasecmp(substr($key,0,4),'HTTP') == 0)
            {
                $header[] = substr($key,5).': '.$value;
            }
        }
        if (isset($_SERVER['PHP_AUTH_DIGEST'])) { 
            $header[] = 'AUTHORIZATION: '.$_SERVER['PHP_AUTH_DIGEST']; 
        } else if (isset($_SERVER['PHP_AUTH_USER']) && isset($_SERVER['PHP_AUTH_PW'])) {
            $header[] = 'AUTHORIZATION: '.base64_encode($_SERVER['PHP_AUTH_USER'] . ':' . $_SERVER['PHP_AUTH_PW']); 
        }
        if (isset($_SERVER['CONTENT_LENGTH'])) { 
            $header[] = 'CONTENT-LENGTH: '.$_SERVER['CONTENT_LENGTH']; 
        }
        if (isset($_SERVER['CONTENT_TYPE'])) { 
            $header[] = 'CONTENT_TYPE: '.$_SERVER['CONTENT_TYPE']; 
        }
    }
    $url = $_SERVER['REQUEST_URI'];
    $host = $_SERVER['HTTP_HOST'];
    $domain = $_SERVER['REQUEST_SCHEME'] . '://' . gethostbyname2($host);

    $curl_opts = array(
        CURLOPT_URL => $domain.$url,
        CURLOPT_CONNECTTIMEOUT => 10,
        CURLOPT_TIMEOUT => 10,
        CURLOPT_AUTOREFERER => true,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_BINARYTRANSFER => true,
        CURLOPT_HEADERFUNCTION => 'header_function',
        CURLOPT_WRITEFUNCTION => 'write_function',
        CURLOPT_CUSTOMREQUEST =>$_SERVER['REQUEST_METHOD'],
        CURLOPT_SSL_VERIFYPEER => false,
        CURLOPT_HTTPHEADER => $header,
        CURLOPT_SSL_VERIFYHOST => false
    );

    if ($_SERVER['REQUEST_METHOD']=='POST')//如果是POST就读取POST信息,不支持
    {
        $curl_opts[CURLOPT_POST] = true; 
        $curl_opts[CURLOPT_POSTFIELDS] = file_get_contents('php://input'); 
    }
    $curl = curl_init();
    curl_setopt_array ($curl, $curl_opts);
    empty($debug) || fwrite($debug,"\r\n".date('Y-m-d H:i:s',time())." URL: ".$curl_opts[CURLOPT_URL]."\r\n".$curl_opts[CURLOPT_POSTFIELDS]."\r\n".implode("\r\n",$header)."\r\n\r\n");
    $ret = curl_exec ($curl);
    if ($GLOBALS['chunked']){
        echo "0\r\n\r\n";
    }
    curl_close($curl);
    unset($curl);
}

// 通过命令行获取host addr
function gethostbyname2($host) {
    $command = "dig $host +short";
    $results = shell_exec("$command 2>&1");
    $dns_arr = explode("\n", $results);
    $hostaddr = '';
    foreach ($dns_arr as $ip) {
        $hostaddr = filter_var($ip, FILTER_VALIDATE_IP);
        if (!empty($hostaddr)) {
            break;
        }
    }
    return $hostaddr;
}

$debug = 1;//设为1开启记录.
if ($debug)
{
    $logdir = dirname(__FILE__).'/debug/';
    if (!file_exists($logdir))
        mkdir($logdir);
    $debug = fopen($logdir.$_SERVER['REMOTE_ADDR'].date('_ymd_',time()).'__'.$_SERVER['SERVER_NAME'].".log",'a');
    fwrite($debug,print_r($_SERVER,true));
    $phpContent = @file_get_contents('php://input');
    fwrite($debug,"===========php://input===========\r\n".$phpContent."\r\n======================\r\n");
}
proxy();
empty($debug) || fclose($debug);
