$(function(){
    var statuses = [];
    var page = 1;

    var report_statuses = function(){
        var tpl = $('#tpl_statuses').html();
        var html = Mustache.render(tpl, {statuses: statuses});
        $('.statuses').html(html); 
    };

    var get_status = function(){
        WB2.anyWhere(function(W){
            W.parseCMD("/statuses/friends_timeline.json", 
                function(sResult, bStatus){
                    if (bStatus){
                        _.each(sResult.statuses, function(status){
                            status = status.retweeted_status || status;
                            if (!_.find(statuses, function(x){return x.id == status.id})){
                                statuses.push(status);
                            }
                        });
                        if (statuses.length >= 300){
                            report_statuses();
                        }else {
                            get_status(); 
                        }
                    }
                }, 
                { count: 100, page: page++ }, 
                { method: 'GET' }
            );
        });
    };

    WB2.login(get_status);

    $('.sortby_default').click(function(){
        statuses = _.sortBy(statuses, function(status){return -status.created_at});
        report_statuses();
    });
    $('.sortby_repost').click(function(){
        statuses = _.sortBy(statuses, function(status){return -status.reposts_count});
        report_statuses();
    });
    $('.sortby_comment').click(function(){
        statuses = _.sortBy(statuses, function(status){return -status.comments_count});
        report_statuses();
    });

});
