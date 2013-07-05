angular.module('myApp', ['ngCookies']);
function DemoCtrl($scope, $http, $cookies) {
    $scope.name = decodeURIComponent($cookies.name);

    //获取所有格子
    $http.get('/houses').success(function(houses) {
        _.each(houses, function(house){
            house.name = decodeURIComponent(house.name);
        });
        $scope.houses = houses;
    });

    //修改格子内容
    $scope.editHouse = function(house){
        var text = prompt("请输入您要说的话", house.text);
        if(!text) return;
        $http.put('/houses', 
                  {name: encodeURIComponent(house.name), 
                   text: text, 
                   lastmodified: moment().format("YYYY-MM-DD mm:ss")}
        ).success(function(){
            house.text = text;
            house.lastmodified = moment().format("YYYY-MM-DD mm:ss");
        });
    };

    //添加格子
    $scope.addHouse = function(){
        var name = prompt("请输入您的姓名", '');
        var text = prompt("请输入您现在要说的话", '');
        name = name && name.trim && name.trim(); 
        name = encodeURIComponent(name);
        if (!name || !text) return;
        var house = {
            name: name,
            text: text,
            lastmodified: moment().format("YYYY-MM-DD mm:ss") 
        };
        $http.post('/houses', house).success(function(){
            house.name = decodeURIComponent(house.name);
            $scope.houses.push(house);
            $scope.name = decodeURIComponent(house.name);
        });
    };

    $scope.willshowcreate = function(){
        return !_.find($scope.houses, function(house){ 
            return house.name == $scope.name;
        }); 
    };
}
