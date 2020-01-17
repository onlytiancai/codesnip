/*
<div id="node-input" class="rectangle-size">数据源</div>
<div id="node-datatable" class="rectangle-size" style="top:200px;">数据显示</div>
<div id="node-pca" class="rectangle-size" style="left: 200px;">主成分分析</div>
<div id="node-datatable2" class="rectangle-size" style="left: 400px;">数据显示</div>

*/

exports.data = [
    {
        type: 'datasource',       
        text: '数据源',
        id: 'node-input'
    },
    {
        type: 'datatable',
        top: '200',        
        text: '数据显示',
        id: 'node-input'
    },
    {
        type: 'pca',        
        left: '200',
        text: '主成分分析',
        id: 'node-input'
    },
    {
        type: 'datatable',
        left: '400',        
        text: '数据显示',
        id: 'node-datatable2'
    },
];