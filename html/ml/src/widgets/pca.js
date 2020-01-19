const type =  'pca';
const text =  '主成分分析';


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