const type =  'datatable';
const text =  '数据表格';


function Node(id, data) {
    this.id = id;
    this.type = type;
    this.text = text;
    this.data = data;
    
    $('#'+id).dblclick(this.dblclick.bind(this));
}

Node.prototype.dblclick = function() {
    console.log('dbclick', this.type, this.text);
}

module.exports = {
    type: type,
    text: text,
    Node: Node,
};

require('jsgrid')
require('jsgrid/dist/jsgrid.css')
require('jsgrid/dist/jsgrid-theme.css')

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