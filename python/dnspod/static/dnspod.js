define(function(require, exports, module) {
    var Backbone = require('backbone');
    var _ = require('underscore');
    var Mustache = require('mustache');

    var tpl = require('./index.html');
    $('.main-body').html(tpl);

    var txt_input = '';

    var check_login = function(callback){
        $.getJSON('/login/userinfo', function(json){
            var usertype = json.data ? json.data.usertype : null;
            if (usertype == 'dnspod'){
                callback(); 
            }else{
                $('.login_container').html("<a href='/login/dnspodlogin'>请先用DNSPod登陆</a>");
            }
        });
    };

    var Step1View = Backbone.View.extend({
        initialize: function(options){
            _.bindAll(this, 'render');
            _.bindAll(this, 'next_step');
        },
        el: $(".step_container"),
        template: require('./step1.tpl'),
        events: {
            "click .next_step": "next_step",
        },
        next_step: function(){
            txt_input = this.$(".txt_input").val();
            location.href = "#step2"
        },
        render: function(){
            $(this.el).html(this.template);
            if (txt_input){
                this.$(".txt_input").val(txt_input);
            }
        }
    });
    var step1_view = new Step1View(); 

    var Step2View = Backbone.View.extend({
        initialize: function(options){
            _.bindAll(this, 'render');
            _.bindAll(this, 'import_record');
        },
        el: $(".step_container"),
        template: require('./step2.tpl'),
        events: {
            "click .import_record": "import_record",
            "click .back": "back",
        },
        back: function(){
            location.href = '#';
        },
        import_record: function(){
            var domain_id = this.$('.selDomains').val();
            var domain = this.$('.selDomains').find("option:selected").text();
            var records = this.model.records;
            var record_count = records.length;
            if (confirm("您确定要把这" + record_count + "条记录导入到"+ domain+ "下吗？")){
                var wrap = function(record){
                    delete record.state;
                    record.domain_id = domain_id;
                    $('#record_status_' + record.seq).html('正在导入...');
                    $.ajax({
                        type: 'POST',
                        url: 'Api/Record.Create',
                        dataType: 'json',
                        data: record,
                        success: function(data){
                            $('#record_status_' + record.seq).html(data.status.message);
                        },
                        error: function(){
                            $('#record_status_' + record.seq).html("导入失败");
                        }
                    });
                };
                for(var i in records){
                    wrap(records[i]);
                }
            }
        },
        render: function(){
            var html = Mustache.render(this.template, this.model);
            $(this.el).html(html);
        }
    });
    var step2_view = new Step2View(); 


    var Workspace = Backbone.Router.extend({
        routes: {
            "": "step1", 
            "step2": "step2"
        },
        step1: function(name){
            check_login(function(){
                step1_view.render();
            });
        },
        step2: function(name){
            if(!txt_input){
                location.href = "#";
                return;
            }
            check_login(function(){
                var parser = require('./parser');
                var records = parser.parse_txt(txt_input);
                $.ajax({
                    type: 'POST',
                    url: 'Api/Domain.List',
                    dataType: 'json',
                    success: function(data){
                        var domains = data.domains;
                        step2_view.model = {records: records, domains: domains};
                        step2_view.render();
                    } 
                });
            });
        }
    });

    var app = new Workspace();
    Backbone.history.start();
});
