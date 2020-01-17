require('webpack-jquery-ui'); // 拖动需要
const Mustache = require('Mustache')

const widgets = [
    require('./pca'),
    require('./datasource'),
    require('./datatable')
];

exports.widgets = widgets;

exports.initToolbox = function () {
    const tpl = '{{#widgets}}<div class="item" data-type="{{type}}">{{text}}</div>{{/widgets}}';
    const html = Mustache.render(tpl, { widgets: widgets });
    $('#toolbox').append(html);

    // 可拖动
    $('#toolbox .item').draggable({
        helper: 'clone',
        scope: 'ss'
    })
}