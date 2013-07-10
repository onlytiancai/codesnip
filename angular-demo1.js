angular.module('common', []);
angular.module('common').directive('commonErrorMsg', function(){
    return {
        restrict: "A",
        controller: function ($scope, $element, $attrs) {
            $element.css('color', 'red');
            $scope.$on('common.showerror', function (ev, msg) {
                $element.html(msg);
            });
        }
    }
});

var myApp = angular.module('myApp', ['common']);
myApp.directive('fruits', function(fruitsService) {
    return {
        restrict: "E",
        transclude: true,
        replace: true,
        template: '<ul ng-transclude></ul>',
        controller: function ($scope, $element, $attrs) {
            $scope.$on('fruitsService.updated', function () {
                $scope.apple_count = fruitsService.apple_count; 
                $scope.orange_count = fruitsService.orange_count;      
            });
        }
    }
})
.directive('orange', function() {
    return {
        restrict: "E",
        template: '<li>桔子</li>'
    }
})
.directive('apple', function() {
    return {
        restrict: "E",
        template: '<li><a ng-click="show()" href="#">苹果</a></li>',
        link: function(scope, element, attrs, $rootScope) {
            scope.show = function(){
                alert('我是一个苹果');
            }; 
        }
    }
})
.controller('statusCtrl', function($scope, fruitsService) {
    $scope.$on('fruitsService.updated', function () {
        $scope.apple_count = fruitsService.apple_count; 
        $scope.orange_count = fruitsService.orange_count;      
    });    
})
.controller('inputCtrl', function($scope, fruitsService, $rootScope) {
    $scope.$watch('apple_count', function (newVal, oldVal, $scope) {
        if (newVal > 10){
            $rootScope.$emit('common.showerror', 'too big');
        }else{
            fruitsService.set_apple_count(newVal);
        }
    }, true);
    $scope.$watch('orange_count', function (newVal, oldVal, $scope) {
        fruitsService.set_orange_count(newVal);
    }, true);
    fruitsService.set_apple_count(3);
    fruitsService.set_orange_count(2);
})
.filter('range', function() {
    return function(input, total) {
        total = parseInt(total);
        for (var i=0; i<total; i++)
            input.push(i);
        return input;
    };
})
.service('fruitsService', function ($rootScope) {
    this.set_apple_count = function (apple_count) {
        this.apple_count = apple_count;
        $rootScope.$broadcast('fruitsService.updated');
    };
    this.set_orange_count = function (orange_count) {
        this.orange_count = orange_count;
        $rootScope.$broadcast('fruitsService.updated');
    };
});
