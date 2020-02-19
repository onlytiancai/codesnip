const dataset = require('ml-dataset-iris').getNumbers();
const fields = ['sepal length', 'sepal width', 'petal length', 'petal width'];

/* global jsPlumb */
function makeStyle(flag) {
    let config = {};
    config.connectorPaintStyle = {
        lineWidth: 1,
        strokeStyle: flag == 'dim' ? '#a2a1a1' : 'black',
        joinstyle: 'round',
        outlineColor: '',
        outlineWidth: ''
    };

    // 鼠标悬浮在连接线上的样式
    config.connectorHoverStyle = {
        lineWidth: 2,
        strokeStyle: '#4caf50',
        outlineWidth: 10,
        outlineColor: ''
    };

    return {
        // 端点形状
        endpoint: ['Dot', {
            radius: 6,
            fill: flag == 'dim' ? '#a2a1a1' : 'black'
        }],
        // 连接线的样式
        connectorStyle: config.connectorPaintStyle,
        connectorHoverStyle: config.connectorHoverStyle,
        // 端点的样式
        paintStyle: {
            fillStyle: flag == 'dim' ? '#a2a1a1' : 'black',
            radius: 4
        },
        hoverPaintStyle: {
            fillStyle: '#4caf50',
            strokeStyle: '#4caf50'
        },
        isSource: true,
        connector: ['Straight', {
            gap: 0,
            cornerRadius: 5,
            alwaysRespectStubs: true
        }],
        isTarget: true,
        // 设置连接点最多可以链接几条线
        maxConnections: -1,
        connectorOverlays: [
            ['Arrow', {
                width: 8,
                length: 10,
                location: 1
            }]
        ]
    };
}

let config = {
    baseStyle: makeStyle('base'),
    dimStyle: makeStyle('dim')
};

jsPlumb.ready(function () {

    let pointStyle = config.baseStyle;

    jsPlumb.setContainer('workspace');
    jsPlumb.addEndpoint('node-input', { uuid: 'node-input', anchor: 'Bottom' }, pointStyle);
    jsPlumb.addEndpoint('node-input', { uuid: 'node-input2', anchor: 'Right' }, pointStyle);
    jsPlumb.addEndpoint('node-datatable', { uuid: 'node-datatable', anchor: 'Top' }, pointStyle);
    jsPlumb.addEndpoint('node-pca', { uuid: 'node-pca', anchor: 'Left' }, pointStyle);
    jsPlumb.addEndpoint('node-pca', { uuid: 'node-pca2', anchor: 'Right' }, pointStyle);
    jsPlumb.addEndpoint('node-datatable2', { uuid: 'node-datatable2', anchor: 'Left' }, pointStyle);

    jsPlumb.connect({ uuids: ['node-input', 'node-datatable'] });
    jsPlumb.connect({ uuids: ['node-input2', 'node-pca'] });
    jsPlumb.connect({ uuids: ['node-pca2', 'node-datatable2'] });

    jsPlumb.draggable(
        $(".rectangle-size,.circle-size"),
        { containment: 'workspace' }
    );

    jsPlumb.bind('click', function (conn, originalEvent) {
        if (window.prompt('确定删除所点击的链接吗？ 输入1确定') === '1') {
            jsPlumb.detach(conn)
        }
    })

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

        const pca = new ML.PCA(dataset);
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


    $('#toolbox .item').draggable({
        helper: 'clone',
        scope: 'ss'
    })

    $('#workspace').droppable({
        scope: 'ss',
        drop: function (event, ui) {
            dropNode(ui.draggable[0], ui.position)
        }
    })

    // 放入拖动节点
    function dropNode(node, position) {        
        position.left -= $('#toolbox').outerWidth()
        position.id = uuid.v1()
        position.type = node.dataset.type;
        position.text = node.innerText

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

});