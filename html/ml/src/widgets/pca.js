const { PCA } = require('ml-pca');

const type =  'pca';
const text =  'PCA';


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

    const {dataset, fields } = this.sourceWidget.getDataset();

    const pca = new PCA(dataset);

    return {
        dataset: [pca.getExplainedVariance()],
        fields: fields,
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
