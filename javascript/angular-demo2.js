angular.module('app', ['ngResource'])
.config(['$httpProvider', function($httpProvider){
    $httpProvider.defaults.headers.post['Content-Type'] = "application/x-www-form-urlencoded;charset=utf-8";
    $httpProvider.defaults.transformRequest = function(data){
        if (data === undefined) {
            return data;
        }
        return $.param(data);
    }
}]);
function MyCtrl($resource) {
    var User = $resource('/Api/:apiname', {apiname:'@apiname'});
    User.save({apiname: 'Domain.List', user_id: 123});
}
