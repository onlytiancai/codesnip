const testdata = require('./testdata').data;
const widgets = require('./widgets/index');
const workflow = require('./workflow');

require('bootstrap3/dist/css/bootstrap.min.css')
require('bootstrap3')
require('bootstrap-filestyle')


$(function () {
    widgets.initToolbox();
    $('#btn-new').click(workflow.new);
    $('#btn-save').click(workflow.save);    
    $('#btn-load').change(function () { workflow.load('btn-load') });
});

jsPlumb.ready(function () {
    workflow.init();
});
