const uuid = require('uuid');
const Mustache = require('Mustache')
const jsPlumb = require('jsplumb').jsPlumb;
const testdata = require('testdata').data;

// 放入拖动节点
function dropNode(position) {
    position.left -= $('#toolbox').outerWidth()
    position.id = uuid.v1()

    var html = renderHtml(position)
    $('#workspace').append(html)

    addDraggable(position.id)
}

// 渲染html
function renderHtml(position) {
    return Mustache.render($('#tpl-node').html(), position)
}

// 让元素可拖动
function addDraggable(id) {
    jsPlumb.draggable(id, {
        containment: 'parent'
    })
}

exports.init = function () {

    jsPlumb.setContainer('workspace');

    jsPlumb.importDefaults({
        Connector: ["Bezier", { curviness: 30 }],
        Endpoint: ["Dot", { radius: 5 }],
        EndpointStyle: { fill: "#567567" },
        Anchor: [0.5, 0.5, 1, 1]
    });

    var endpointOptions = { isSource: true, isTarget: true };
    var nodeInputEp1 = jsPlumb.addEndpoint('node-input', { anchor: "Bottom" }, endpointOptions);
    var nodeInputEp2 = jsPlumb.addEndpoint('node-input', { anchor: "Right" }, endpointOptions);

    var nodeDatatableEp1Top = jsPlumb.addEndpoint('node-datatable', { anchor: "Top" }, endpointOptions);

    var nodePCAEp1 = jsPlumb.addEndpoint('node-pca', { anchor: "Left" }, endpointOptions);
    var nodePCAEp2 = jsPlumb.addEndpoint('node-pca', { anchor: "Right" }, endpointOptions);

    var nodeDatatableEp2Left = jsPlumb.addEndpoint('node-datatable2', { anchor: "Left" }, endpointOptions);

    jsPlumb.connect({ source: nodeInputEp2, target: nodePCAEp1 });
    jsPlumb.connect({ source: nodeInputEp1, target: nodeDatatableEp1Top });
    jsPlumb.connect({ source: nodePCAEp2, target: nodeDatatableEp2Left });


    jsPlumb.draggable(
        $(".rectangle-size,.circle-size"),
        { containment: 'workspace' }
    );

    jsPlumb.bind('click', function (conn, originalEvent) {
        if (window.prompt('确定删除所点击的链接吗？ 输入1确定') === '1') {
            jsPlumb.detach(conn)
        }
    })
}


exports.droppable = function () {

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



}