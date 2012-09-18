$(function(){
    var assert = {
        assertEqual: function(msg, s1, s2){
            var html = msg + ':' + (s1 === s2 ? 'ok': 'faild');
            $('#result').html(html);
        }   
    };

    $.each(wawatests, function(){
        var that = this;
        $('#frm_test').load(function(){
            var test_frm = $(this).contents();
            that.action(test_frm);
            setTimeout(function(){
                that.callback(test_frm, assert);
            }, 1000);
        });
        $('#frm_test').attr('src', that.url);
    });
}); 

