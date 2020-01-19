const uuid = require('uuid');
const Mustache = require('Mustache')
const jsPlumb = require('jsplumb').jsPlumb;
var FileSaver = require('file-saver');
const widgets = require('./widgets/index');

const srcOptions = { isSource: true, anchor: ["Perimeter", { shape: "Rectangle" }], maxConnections: -1 };
const dstOptions = { isTarget: true, anchor: "Left", endpoint: ["Rectangle", { width: 8, height: 8 }] };
const nodeTpl = $('#tpl-node').html();

const utils = require('./utils')

// 全局 nodes，保存时用， connects, 位置不实时更新
let allNodes = [];

// 拖入节点
function dropNode(position) {
    position.left -= $('#toolbox').outerWidth()
    position.id = uuid.v1()
    position.connects = [];

    addNode(position);
}

// 添加节点
function addNode(options) {
    var html = Mustache.render(nodeTpl, options)
    $('#workspace').append(html)
    
    // 添加到全局 nodes    
    const node = widgets.makeNode(options)    
    allNodes.push(node);
    
    // 添加端点               
    jsPlumb.addEndpoint(node.id, { uuid: node.id + '-source', }, srcOptions);
    jsPlumb.addEndpoint(node.id, { uuid: node.id + '-target', }, dstOptions);

    // 设置可拖动
    jsPlumb.draggable(node.id, { containment: 'workspace' })
}

// 清空所有
function clearAll() {
    allNodes = [];
    jsPlumb.empty("workspace");
    $('#workspace').empty();
}

exports.new = function () {
    if (utils.confirm("新建工作流会清空现有数据，确认要清空吗？")) {
        clearAll();
    }
};

exports.save = function () {

    if (allNodes.length == 0) {
        utils.showWarning('空白文件，无法保存。');
        return;
    }

    // 保存位置
    for (const node of allNodes) {
        const p = $('#'+node.id).position();
        node.data.left = p.left;
        node.data.top = p.top;
    }

    // 清空 connect ，重新添加
    allNodes.forEach(n => n.connects = []);

    // 遍历 connect 添加到 node
    const connects = jsPlumb.select();
    connects.each(function (c) {
        const found = allNodes.find(n => n.id == c.sourceId);
        found.connects.push(c.targetId);
    });

    var blob = new Blob([JSON.stringify(allNodes, null, 2)], { type: "text/plain;charset=utf-8" });
    FileSaver.saveAs(blob, "workflow.json");

}

// 加载 workflow
exports.load = function (fileInput) {
    
    var selectedFile = document.getElementById(fileInput).files[0];
    var reader = new FileReader();
    reader.readAsText(selectedFile);
    reader.onload = function () {        
        console.debug("workflow onload", selectedFile.name,  selectedFile.size);
        clearAll();
    
        let input = JSON.parse(this.result);

        // 添加节点
        for (const node of input) {
            addNode(node.data);
        }

        // 连线
        input.forEach(function (n) {
            n.connects.forEach(function (c) {
                jsPlumb.connect({ uuids: [n.id + '-source', c + '-target'] });
            });
        });
    };


}

// 工作区初始化
exports.init = function () {

    // 设置工作区
    jsPlumb.setContainer('workspace');

    // 设置默认工作流默认配置
    jsPlumb.importDefaults({
        Connector: ["Bezier", { curviness: 30 }],
        Endpoint: ["Dot", { radius: 5 }],
        EndpointStyle: { fill: "#567567" },
        Anchor: [0.5, 0.5, 1, 1]
    });

    // 设置 widget 可放置区域
    $('#workspace').droppable({
        scope: 'ss',
        drop: function (event, ui) {
            const node = ui.draggable[0];
            const position = ui.position;
            position.type = node.dataset.type;
            position.text = node.innerText;
            dropNode(position);
        }
    })

    // 点击锚点可删除连线
    jsPlumb.bind('click', function (conn, originalEvent) {
        if (window.prompt('确定删除所点击的链接吗？ 输入1确定') === '1') {
            jsPlumb.detach(conn)
        }
    })

}