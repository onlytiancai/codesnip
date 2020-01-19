const dataset = require('ml-dataset-iris').getNumbers();

const type =  'datasource';
const text =  '数据源';


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