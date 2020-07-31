<?php
// https://github.com/open-scratch/scratch3.git
ini_set("display_errors", "On");

// 作品上传目录
const CREATION_DIR = 'creations';

// 简单路由，根据 action 调用函数
if (!empty($_REQUEST['action'])) {
    $action = strtolower($_REQUEST['action']);
    if (!is_callable($action)) exit("404");    
    call_user_func($action);
    die();
}


// 作品列表
function creation_list()
{
    $files = scandir(CREATION_DIR);
    foreach ($files as $file ) {
        if ($file == '..' || $file == '.') continue;
        echo '<a href="?sb='.urlencode($file).'">'.htmlspecialchars($file).'</a><br>';
    }
}

// 作品上传
function upload_sb()
{
    if ($_SERVER['REQUEST_METHOD'] == 'POST') {
        if ($_FILES["upfile"]["error"] > 0) {
            exit("error：" . $_FILES["upfile"]["error"]);
        }
        else {
            if (!file_exists(CREATION_DIR)) {
                if (!is_writable('.')) exit('Permission denied to mkdir');
                mkdir(CREATION_DIR);            
            } 

            $file_name = CREATION_DIR . DIRECTORY_SEPARATOR. $_FILES['upfile']['name'];
            if (!is_writable(CREATION_DIR)) exit('Permission denied to move_uploaded_file');
            if (file_exists($file_name)) exit($file_name." 文件已经存在。");
           
            move_uploaded_file($_FILES["upfile"]["tmp_name"], $file_name);
            exit("ok");
           
        }
    }
    
}
?>
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="google" value="notranslate">
  <link rel="shortcut icon" href="static/favicon.ico">
  <title>Action Scratch</title>
  <script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.5.1/jquery.js"></script>
  <script>
    function getParameterByName(name) {
      var match = RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
      return match && decodeURIComponent(match[1].replace(/\+/g, ' '));
    }
    window.scratchConfig = {
      logo: {
        show: true
        // , url: "data:image/png;base64,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"
        , url: "static/action-logo.png"
        , handleClickLogo: () => {
          console.log('点击LOGO')
        }
      },
      menuBar: {
        color: '#0DA8E9'
      },
      shareButton: {
        show: true,
        buttonName: "保存",
        handleClick: () => {
          //点击分享按钮
          console.log('分享按钮')
          window.scratch.getProjectCover(cover => {
            //TODO 获取到作品截图
            console.log(cover)
          })

          window.scratch.getProjectFile(file => {

            // 获取到项目名
            var projectName = window.scratch.getProjectName()
            console.log(projectName);

            var form = new FormData();
            form.append("upfile", file, projectName + ".sb3");
            $.ajax({
              url: '?action=upload_sb'
              , type: "POST"
              , contentType: false //让xhr自动处理Content-Type header，multipart/form-data需要生成随机的boundary
              , processData: false //不要处理data，让xhr自动处理
              , data: form
              , success: function (v) {
                alert("上传成功:" + v);
              }
              , error: function (s) {
                alert("上传失败:" + s);
              }
            });
          })
        }
      },
      profileButton: {
        show: true,
        buttonName: "我的作品",
        handleClick: () => {
          location.href = '?action=creation_list'
          //点击profile按钮
        }
      },
      stageArea: {
        showControl: false, //是否显示舞台区控制按钮
        showLoading: false, //是否显示Loading
      },
      handleVmInitialized: (vm) => {
        window.vm = vm
        console.log("VM初始化完毕")

      },
      handleProjectLoaded: () => {
        console.log("作品载入完毕")

      },
      handleDefaultProjectLoaded: () => {
        //默认作品加载完毕，一般在这里控制项目加载
        // window.scratch.setProjectName("默认项目")
        //  window.scratch.loadProject("/project.sb3", () => { 
        //     console.log("项目加载完毕")
        //     window.scratch.setProjectName("默认项目")
        //  })
      },
      //默认项目地址,不需要修请删除本配置项
      //defaultProjectURL: "default.sb3",
      //若使用官方素材库请删除本配置项, 默认为/static下的素材库
      assetCDN: './static'
    }

    const CREATION_DIR = 'creations';
    const sb = getParameterByName('sb');
    if (sb) scratchConfig.defaultProjectURL = CREATION_DIR + '/' + sb;


  </script>
</head>

<body>
  <div id="scratch">

  </div>
  <script type="text/javascript" src="lib.min.js"></script>
  <script type="text/javascript" src="chunks/gui.js"></script>
</body>
<script>

</script>

</html>
