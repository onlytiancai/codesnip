import Vue from "vue";
import App from './scatterchart.vue';
const AppClass = Vue.extend(App);

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
    console.log(111, classes, classLabels);
    $('#detail-box').html('<div class="chart-tools"></div><div id="main" style="width:100%;height:500px"></div>');
    this.vue = new AppClass({ data: {
        dataset: dataset,
        fields: fields,
        classes: classes,
        classLabels: classLabels,
        xAxis: 0,
        yAxis: 1,
    }});

    $('#detail-box .chart-tools').append(this.vue.$mount().$el);
}

   