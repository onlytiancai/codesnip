const testdata = require('./testdata').data;
const widgets = require('./widgets/index');
const workflow = require('./workflow');

$(function () {
    widgets.initToolbox();
    $('#btn-save').click(workflow.save);    
    $('#btn-load').change(function () { workflow.load('btn-load') });
});

jsPlumb.ready(function () {
    workflow.init();
});
