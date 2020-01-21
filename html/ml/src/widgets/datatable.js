import Vue from "vue";
import App from './datatable.vue';
const AppClass = Vue.extend(App);

const type = 'datatable';
const text = '数据表格';


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


// 分页获取数据
Node.prototype.fetch = function (offset, limit) {
    return this.dataset.slice(offset, offset + limit);
}

// 双击
Node.prototype.dblclick = function () {
    if (!this.sourceWidget) {
        $('#detail-box').html('未设置数据源');
        return;
    }

    const {dataset, fields } = this.sourceWidget.getDataset();
    
    this.columns = fields.map(function (x) {
        return { title: x, field: x, sortable: true }
    });

    // [1,2,3,4] => {a:1, b:2, c:3, d:4}
    this.dataset = dataset.map(x => x.reduce(function (pre, cur, i) { pre[fields[i]] = cur; return pre }, {}));

    const vueData = {
        columns: this.columns,
        data: [],
        total: this.dataset.length,
        query: {},
        widget: this,
    }

    this.vue = new AppClass({ data: vueData });

    $('#detail-box').empty().append(this.vue.$mount().$el);
    console.log('dbclick', this.type, this.text);
}

module.exports = {
    type: type,
    text: text,
    Node: Node,
};

const ml = require('ml');

$('#node-datatable').click(function () {
    const data = dataset.map(x => x.reduce(function (pre, cur, i) { pre[fields[i]] = cur; return pre }, {}));

    $("#grid").jsGrid({
        width: "100%",
        height: "600px",
        data: data,
        fields: fields.map(function (x) { return { name: x } }),
    });
});

$('#node-datatable2').click(function () {

    const pca = new ml.PCA(dataset);
    console.log(pca.getExplainedVariance());

    const newPoints = [[4.9, 3.2, 1.2, 0.4], [5.4, 3.3, 1.4, 0.9], [4.6, 3.4, 1.4, 0.3]];
    const ret = pca.predict(newPoints);
    console.log(ret);
    const data = ret.toJSON().map(x => x.reduce(function (pre, cur, i) { pre[fields[i]] = cur; return pre }, {}));

    $("#grid").jsGrid({
        width: "100%",
        height: "600px",
        data: data,
        fields: fields.map(function (x) { return { name: x } }),
    });
});