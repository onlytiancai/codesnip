<?php
ini_set("display_errors", "On");
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    if ($_FILES["upfile"]["error"] > 0) {
        exit("error：" . $_FILES["upfile"]["error"]);
    }
    else {
        $file_name = $_FILES['upfile']['name'];
        $new_name = date('YmdHis').rand(10000, 99999).'.'.pathinfo($file_name, PATHINFO_EXTENSION);
        if (file_exists($new_name)) {
            exit($new_name." 文件已经存在。");
        }
        else {
            move_uploaded_file($_FILES["upfile"]["tmp_name"], $new_name);
            exit("ok");
        }
    }
}
?>
<!DOCTYPE html>
<html lang="zh-cn">
    <head>
        <meta charset="utf-8">
        <title>Voice Recorder</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/gh/xiangyuecn/Recorder@latest/recorder.mp3.min.js"></script>
        <script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    </head>
    <body>
        <p id="msg"></p>
        <button onclick="start_record()" id="start">开始录音</button>
        <button onclick="upload_record()" id="upload">停止并上传</button>
<script>
var TestApi="?action=upload";//用来在控制台network中能看到请求数据，测试的请求结果无关紧要
var rec=Recorder();

$("#start").attr({"disabled":"disabled"}); 
$("#upload").attr({"disabled":"disabled"}); 

function start_record() {
    $('#msg').text("开始录音");
    $("#start").attr({"disabled":"disabled"}); 
    $('#upload').removeAttr("disabled");
    rec.start();
}

function upload_record() {
    $("#start").attr({"disabled":"disabled"}); 
    $("#upload").attr({"disabled":"disabled"}); 
    $('#msg').text("停止并上传");

    rec.stop(function(blob,duration){
        //-----↓↓↓以下才是主要代码↓↓↓-------

        //本例子假设使用jQuery封装的请求方式，实际使用中自行调整为自己的请求方式
        //录音结束时拿到了blob文件对象，可以用FileReader读取出内容，或者用FormData上传
        var api=TestApi;

        /***方式二：使用FormData用multipart/form-data表单上传文件***/
        var form=new FormData();
        form.append("upfile",blob,"recorder.mp3"); //和普通form表单并无二致，后端接收到upfile参数的文件，文件名为recorder.mp3
        //...其他表单参数
        $.ajax({
        url:api //上传接口地址
            ,type:"POST"
            ,contentType:false //让xhr自动处理Content-Type header，multipart/form-data需要生成随机的boundary
            ,processData:false //不要处理data，让xhr自动处理
            ,data:form
            ,success:function(v){
                $('#start').removeAttr("disabled");
                $('#msg').text("上传成功:"+v);
                location.reload();
            }
        ,error:function(s){
            $('#start').removeAttr("disabled");
            $('#msg').text("上传失败:"+s);
        }
        });

        //-----↑↑↑以上才是主要代码↑↑↑-------
    }
    ,function(msg){
        $('#start').removeAttr("disabled");
        $('#msg').text("录音失败:"+msg);
    });

}


rec.open(function(){
    $('#start').removeAttr("disabled");
}, function(msg){
    $('#msg').text("无法录音:"+msg);
});
</script>

<ul>
    <?php
    $list = glob('*.mp3');
    foreach($list as $mp3):
    ?>
    <li>
        <?= $mp3 ?> <audio src="<?= $mp3?>" controls> </audio> <br>
    </li>
    <?php endforeach?>
</ul>

    </body>
</html>
