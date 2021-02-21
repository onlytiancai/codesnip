<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0">
    <title>基金分析</title>
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
        <div class="col-md-12">
            <h2>Hello World</h2>
            <div class="control-group">
                <select id="sel-funds" class="demo-default" placeholder="请输入基金，可多选..." multiple="multiple">
                    <option value="">请输入基金，可多选...</option>
                </select>
            </div>
            <button type="button" class="btn btn-default" id="check">分析</button>
            <div id="main" style="height:400px;"></div>
        </div>
      </div>
    </div>
    <script>
    $(document).ready(function() {
      $('#sel-funds').selectize({
        valueField: 'code',
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
        },

        render:{
          option: function(item,escape) {
            return '<div class="option">【' + item.code + '】' + item.name + '</div>';
          },
          item:function(item) {
		    return '<div>【' + item.code + "】" + item.name + '</div>';
          }
        },
      });

      function showPosition(labels, data)
      {
        
        var myChart = echarts.init(document.getElementById('main'));

        // 指定图表的配置项和数据
        var option = {
            tooltip: {},
            legend: {
                data:['仓位']
            },
            xAxis: {
                data: labels 
            },
            yAxis: {},
            series: [{
              itemStyle: {
                normal: {
                  label: {
                    show: true,
                    position: 'top',
                    formatter: function (a, b, c) {
                      return a.data.toFixed(2) + "%";
                    }
                  }
                }
              },
              name: '仓位',
              type: 'bar',
              data: data 
            }]
        };

        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
      }

      var sel = $('#sel-funds')[0].selectize;
      $("#check").click(function() {
        var ajaxs = sel.getValue().map(function(code){
          return $.getJSON('action.php?action=getPosition&code='+code);
        });
        $.when.apply($, ajaxs).done(function(){
          var positionMap = {};
          var rsps= ajaxs.length > 1 ? Array.prototype.slice.call(arguments): [arguments];
          console.log(rsps);
          rsps.forEach(function(rsp){
            var stockList = rsp[0] && rsp[0].stockList || [];
            stockList.forEach(function(stock){
              var name = stock[1], percent=parseFloat(stock[2]);
              positionMap[name] = (positionMap[name] || 0) + percent;
            });
          });
          var positions = Object.entries(positionMap).sort(function(x,y){
            return y[1]-x[1];
          }).slice(0,20);
          console.log(positions);
          showPosition(positions.map(function(x){
            return x[0];
          }), positions.map(function(x){
            return x[1]/ajaxs.length; 
          }));
        });
	  });
    });
    </script>



    <script src="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/js/bootstrap.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.0.2/dist/echarts.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/selectize.js/0.12.6/js/standalone/selectize.min.js"></script>
    <!-- 为ECharts准备一个具备大小（宽高）的Dom -->
    <script type="text/javascript">
        // 基于准备好的dom，初始化echarts实例
    </script>

  </body>
</html>
