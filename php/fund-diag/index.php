<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Bootstrap 101 Template</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/css/bootstrap.min.css" rel="stylesheet">
    <!--[if lt IE 9]>
      <script src="https://cdn.jsdelivr.net/npm/html5shiv@3.7.3/dist/html5shiv.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/respond.js@1.4.2/dest/respond.min.js"></script>
    <![endif]-->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/selectize.js/0.12.6/css/selectize.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/jquery@1.12.4/dist/jquery.min.js"></script>
  </head>
  <body>

    <div class="container">
      <div class="row">
        <div class="col-md-6">
            <div class="control-group">
                <select id="select-yourself" class="demo-default" placeholder="请输入基金..." multiple="multiple">
                    <option value="">请输入基金...</option>
                </select>
            </div>
        </div>
      </div>
    </div>
    <script>
      $(document).ready(function() {

$('#select-yourself').selectize({
  valueField: 'name',
  labelField: 'name',
  searchField: 'name',
  options: [],
  create: false,
  load: function(query, callback) {
    if (!query.length) return callback();
    $.ajax({
      url: 'action.php',
      type: 'GET',
      dataType: 'json',
      data: {
        action: 'searchFund',
        keyword: query,
      },
      error: function() {
        callback();
      },
      success: function(res) {
        callback(res);
      }
    });
  }
});
        });
    </script>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/js/bootstrap.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/selectize.js/0.12.6/js/standalone/selectize.min.js"></script>
  </body>
</html>
