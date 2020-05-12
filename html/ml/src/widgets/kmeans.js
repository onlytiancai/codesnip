const kmeans = require('ml-kmeans').default;

const type =  'kmeans';
const text =  'K-means';


function Node(id, data) {
    this.id = id;
    this.type = type;
    this.text = text;
    this.data = data;
    
    $('#'+id).dblclick(this.dblclick.bind(this));
}

Node.prototype.dblclick = function() {
    if (!this.sourceWidget) {
        $('#detail-box').html(this.type + ':未设置数据源');
        return;
    }

    $('#detail-box').html('已设置数据源' + this.sourceWidget.type + this.sourceWidget.id);
 
}

// 获取数据集，数据表组件获取数据用
Node.prototype.getDataset = function () {
    console.debug('get dataset:', this.id, this.type);

    if (!this.sourceWidget) {
        $('#detail-box').html(this.type + ':未设置数据源');
        return {};
    }

    const ret = this.sourceWidget.getDataset();
    if (!ret.fields) {
        $('#detail-box').html('数据源为空');
        return;
    }

    const {dataset, fields, classes, classLabels } = ret;

    // TODO: k 可设置
    const k = 3;
    let ans = kmeans(dataset, k, { initialization: 'random' });
    console.log(111, ans);

    let range = (start, end) => new Array(end - start).fill(start).map((el, i) => start + i);
    
    return {
        dataset: dataset,
        fields: ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'],
        classes: ans.clusters,
        classLabels: range(0, k),
    }
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
