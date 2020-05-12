var echarts = require('echarts');

const type = 'barchart';
const text = '柱状图';


function Node(id, data) {
    this.id = id;
    this.type = type;
    this.text = text;
    this.data = data;

    $('#' + id).dblclick(this.dblclick.bind(this));
}

module.exports = {
    type: type,
    text: text,
    Node: Node,
};

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
    const {dataset, fields } = ret;

    $('#detail-box').html('<div class="chart-container" style="width:100%;height:500px"></div>');
    var myChart = echarts.init($('#detail-box .chart-container')[0]);
    myChart.setOption({
        title: {
            text: this.sourceWidget.text + '结果显示'
        },
        tooltip: {},
        xAxis: {
            data: fields
        },
        yAxis: {},
        series: [{
            name: '销量',
            type: 'bar',
            data: dataset[0]
        }]
    });
}