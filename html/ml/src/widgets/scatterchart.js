const type = 'scatterchart';
const text = '散点图';

function Node(id, data) {
    this.id = id;
    this.type = type;
    this.text = text;
    this.data = data;

    $('#' + id).dblclick(this.dblclick.bind(this));
}

// 设置数据源
Node.prototype.setSource = function (widget) {
    console.debug('set source', this.id, this.type, widget.id, widget.type);
    this.sourceWidget = widget;
}

// 清空数据源
Node.prototype.clearSource = function () {
    console.debug('clear source', this.id, this.type);
    this.sourceWidget = null;
}

module.exports = {
    type: type,
    text: text,
    Node: Node,
};

// 双击
Node.prototype.dblclick = function () {

    if (!this.sourceWidget) {
        $('#detail-box').html('未设置数据源');
        return;
    }

    const ret = this.sourceWidget.getDataset();
    if (!ret.fields) {
        $('#detail-box').html('数据源为空');
        return;
    }

    const {dataset, fields, classes, classLabels } = ret;
    var echarts = require('echarts');
    var option = {
        xAxis: {},
        yAxis: {},
        series: [{
            data: dataset.map(function (x) {
                return [x[0], x[3]]
            }),
            type: 'scatter',
            itemStyle: {
                normal: {
                    color: function (e) {                        
                        const c = classes[e.dataIndex];
                        const colorIndex = classLabels.indexOf(c);
                        var colorarrays = ["#2F9323","#D9B63A","#2E2AA4","#9F2E61","#4D670C","#BF675F","#1F814A","#357F88","#673509","#310937","#1B9637","#F7393C"];
                        return colorarrays[colorIndex];
                    }
                }
            },
        }]
    };

    $('#detail-box').html('<div class="chart-container" style="width:100%;height:500px"></div>');
    var myChart = echarts.init($('#detail-box .chart-container')[0]);
    myChart.setOption(option);
}

   