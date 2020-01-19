const dataset = require('ml-dataset-iris').getNumbers();
var Vue = require('vue')

const type = 'datasource';
const text = '数据源';

// const App = require('./app.vue')
import App from './app.vue';

function Node(id, data) {
    this.id = id;
    this.type = type;
    this.text = text;
    this.data = data;

    $('#' + id).dblclick(this.dblclick.bind(this));
}

Node.prototype.dblclick = function () {
    new Vue({ el: '#app', template: '<App/>', components: { App } })
    console.log('dbclick', this.type, this.text);
}

module.exports = {
    type: type,
    text: text,
    Node: Node,
};