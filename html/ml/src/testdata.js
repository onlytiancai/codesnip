exports.data = [
    {
        type: 'datasource',       
        text: '数据源',
        top: 10,
        left: 10,
        id: 'n-input',
        connects: ['n-datatable', 'n-pca']
    },
    {
        type: 'datatable',
        top: 200,     
        left: 10,   
        text: '数据显示',
        id: 'n-datatable',
        connects: []
    },
    {
        type: 'pca',        
        top: 10,
        left: 200,
        text: '主成分分析',
        id: 'n-pca',
        connects: ['n-datatable2']
    },
    {
        type: 'datatable',
        top: 10,
        left: 400,        
        text: '数据显示',
        id: 'n-datatable2',
        connects: []
    },
];