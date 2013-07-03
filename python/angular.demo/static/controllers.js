angular.module('myApp', ['ngCookies']);
function DemoCtrl($scope, $http, $cookies) {
    $scope.name = $cookies.name;
    $http.get('/houses').success(function(data) {
        $scope.houses = data;
    });
    $scope.editHouse = function(house){
        var text = prompt("请输入您要说的话", house.text);
        house.text = text;
        $http.put('/houses', house);
    };

    $scope.addHouse = function(){
        var name = prompt("请输入你的姓名", '');
        var text = prompt("请输入您要说的话", '');
        var house = {
            name: name,
            text: text,
            lastmodified: '2013-03-04'
        };
        $http.post('/houses', house).success(function(){
            $scope.houses.push(house);
        });
    };
}
