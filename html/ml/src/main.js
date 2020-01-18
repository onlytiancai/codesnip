const testdata = require('./testdata').data;
const widgets = require('./widgets/index');
const workflow = require('./workflow');

$(function () {    
    widgets.initToolbox();
});

jsPlumb.ready(function () {
    workflow.init();
    workflow.load(testdata);
});
