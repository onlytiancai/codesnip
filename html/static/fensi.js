$(function(){
    var map_mentions = {};
    var add_mentions = function(user){
        var id = user.id;
        if (map_mentions[id] == undefined){
            map_mentions[id] = {};
            map_mentions[id].count = 1;
            map_mentions[id].user = user;
        }
        else{
            map_mentions[id].count += 1;
        }
    };

    var show_users = function(){
        var users = _.values(map_mentions);  
        users = _.sortBy(users, function(user){ return -user.count; });
        _.each(users, function(user){
            console.log(user.user.screen_name, user.count);
            var html = '<li><a href="http://weibo.com/'+user.user.id+'">'+user.user.screen_name+'</a>:'+user.count+'</li>'
            $('#users').append(html);
        });
    };

    var get_mentions = function(W, page){
        W.parseCMD("/statuses/mentions.json", function(sResult, bStatus){
            _.each(sResult.statuses, function(status){
                add_mentions(status.user); 
                var day = moment(status.created_at).format('YYYY MM DD');
            });
            if (page != 4){
                get_mentions(W, ++page);
            }else{
                show_users();
            }
        },
        { count: 200, page: 1 },
        { method: 'GET' });
    };
    WB2.login(function(){
        WB2.anyWhere(function(W){
            get_mentions(W, 1);
        });
    });
});
