const dataset = require('ml-dataset-iris');
const  fields =  ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'];
var Vue = require('vue')

const type = 'datasource';
const text = '数据源';

import App from './datasource.vue';
const AppClass = Vue.extend(App);



const input = {
    name: 'iris',
    data: dataset.getNumbers(),
    cols: fields,
};

const vueDefault = {
    name: input.name,
    cols: input.cols,
    rows: input.data.length,    
}

function Node(id, data) {
    this.id = id;
    this.type = type;
    this.text = text;
    this.data = data;
    
    this.vue = new AppClass({data: this.data.vuedata || vueDefault});    

    $('#' + id).dblclick(this.dblclick.bind(this));
}

Node.prototype.getData = function () {
    return input;
}

// 获取数据集，数据表组件获取数据用
Node.prototype.getDataset = function () {
    console.debug('get dataset:', this.id, this.type);
    return {
        dataset: dataset.getNumbers(),
        fields: ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'],
        classes: dataset.getClasses(),
        classLabels: ['setosa', 'versicolor', 'virginica'],
    }
}

Node.prototype.dblclick = function () {
    $('#detail-box').empty().append(this.vue.$mount().$el);    
    console.log('dbclick', this.type, this.text);
}

module.exports = {
    type: type,
    text: text,
    Node: Node,
};