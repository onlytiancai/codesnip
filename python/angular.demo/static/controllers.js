function HousesCtrl($scope, $http, $cookies) {
    $scope.orderProp = '-lastmodified'; 

    var fech_all_houses = function(){
        $http.get('/houses').success(function(houses) {
            $scope.houses = houses;
            $scope.name = decodeURIComponent($cookies.name);
        });
    };

    fech_all_houses();

    //修改格子内容
    $scope.editHouse = function(house){
        var text = prompt("请输入您要说的话", house.text);
        if(!text) return;
        house.text = text;
        $http.put('/houses', house).success(function(){
            fech_all_houses();
        });
    };

    //添加格子
    $scope.addHouse = function(){
        var name = prompt("请输入您的姓名", '');
        var text = prompt("请输入您现在要说的话", '');
        if (!name || !text) return;
        var house = {
            name: name,
            text: text,
        };
        $http.post('/houses', house).success(function(){
            fech_all_houses();
        });
    };

    $scope.willshowcreate = function(){
        return !_.find($scope.houses, function(house){ 
            return house.name == $scope.name;
        }); 
    };
}

function HistoryCtrl($scope, $http, $routeParams, $cookies) {
    $scope.orderProp = '-lastmodified'; 
    $scope.history_name = $routeParams.name;
    $scope.name = decodeURIComponent($cookies.name);
    $http.get('/history/' + $scope.history_name).success(function(houses) {
        $scope.houses = houses;
    });
}

angular.module('myApp', ['ngCookies']).
config(['$routeProvider', function($routeProvider) {
    $routeProvider.
    when('/houses', {templateUrl: 'static/partials/houses.html', controller: HousesCtrl}).
    when('/house/:name', {templateUrl: 'static/partials/house-history.html', controller: HistoryCtrl}).
    otherwise({redirectTo: '/houses'});
}]);
