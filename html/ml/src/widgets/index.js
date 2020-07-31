require('webpack-jquery-ui'); // 拖动需要
const Mustache = require('Mustache')

const widgets = [
    require('./pca'),
    require('./datasource'),
    require('./datatable'),
    require('./barchart'),
    require('./scatterchart'),
    require('./kmeans'),
];

const widgetMap = widgets.reduce((pre, cur) => { pre[cur.type] = cur; return pre; }, {});

// 根据 type 创建一个新 Node
exports.makeNode = function (options) {
    const widget = widgetMap[options.type];
    if (!widget) throw new Error(`Unknow widget type:${options.type}`);
    return new widget.Node(options.id, options);

}

// 初始化工具箱
exports.initToolbox = function () {
    const tpl = '{{#widgets}}<div class="item" data-type="{{type}}">{{text}}</div>{{/widgets}}';
    const html = Mustache.render(tpl, { widgets: Object.values(widgetMap) });
    $('#toolbox').append(html);

    // 可拖动
    $('#toolbox .item').draggable({
        helper: 'clone',
        scope: 'ss'
    })
}

