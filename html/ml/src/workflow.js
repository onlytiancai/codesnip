const uuid = require('uuid');
const Mustache = require('Mustache')
const jsPlumb = require('jsplumb').jsPlumb;

const srcOptions = { isSource: true, anchor: ["Perimeter", { shape: "Rectangle" }], maxConnections: -1 };
const dstOptions = { isTarget: true, anchor: "Left", endpoint: ["Rectangle", { width: 8, height: 8 }] };
var nodeTpl = $('#tpl-node').html();

// 拖入节点
function dropNode(position) {
    position.left -= $('#toolbox').outerWidth()
    position.id = uuid.v1()

    addNode(position);
}

// 添加节点
function addNode(position) {
    var html = Mustache.render(nodeTpl, position)
    $('#workspace').append(html)

    // 添加端点               
    jsPlumb.addEndpoint(position.id, { uuid: position.id + '-source', }, srcOptions);
    jsPlumb.addEndpoint(position.id, { uuid: position.id + '-target', }, dstOptions);
    // 设置可拖动
    jsPlumb.draggable(position.id, { containment: 'workspace' })
}


// 加载 workflow
exports.load = function (input) {

    // 添加节点
    input.forEach(addNode);

    // 连线
    input.forEach(function (n) {
        n.connects.forEach(function (c) {
            jsPlumb.connect({ uuids: [n.id + '-source', c + '-target'] });
        });
    });
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