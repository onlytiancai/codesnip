/* 
### 源码说明
1. 页面初始化时有3个苹果，3个桔子，用户可以在输入框里重新输入桔子和苹果的数量，界面会有相应的变化
1. 定义了两个模块
    1. common是通用模块，
        1. 包含一个commonErrorMsg的directive用来显示全局的错误信息，
           通过监听common.showerror事件来获取信息，并让字体显示为红色
    1. myApp是整个单页面应用的模块，
        1. 包含inputCtrl, statusCtrl两个controller
        1. 包含fruits, orange, apple三个directive 
        1. 包含range的filter
        1. 包含fruitsService的service
1. 总体依赖关系如下
    1. myApp依赖common
    1. fruits, inputCtrl, statusCtrl都依赖fruitsService
    1. inputCtrl通过事件隐含依赖common
    1. 总体来说上层module依赖底层module，上层controller依赖底层service
1. fruits是一个自定义的directive，用来显示所有水果
    1. transclude=True表示它的子元素也受它管理，比如里面的时苹果和桔子
    1. 该directive要和inputCtrl进行通信，以便动态更改水果的数量，
       所以它和inputCtrl共同依赖fruitsService，并通过fruitsService的事件进行通信。 
1. 事件基本是全局的，所以定义事件时尽量有个命名空间， 如common.showerror, fruitsService.updated
1. orange和apple是两个很普通的directive，其中apple还掩饰了directive里如何处理自身的UI事件
1. statusCtrl就监听fruitsService.updated事件，并更新自己的状态
1. inputCtrl里watch自身UI里的两个ng-model，适时调用fruitsService的相关方法
    1. 如果界面输入太大的数字，会向common.showerror发送消息，以在界面上提示给用户
       这里没有用ng-form自带的验证就是为了演示模块间如何通信
1. range的filter是弥补ng-repeat的不足，让它支持类似 x in range(10)的形式
1. fruitsService纯粹是为了directive之间或controller之间通信和共享数据所设计
 */
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
        link: function(scope, element, attrs) {
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
            $rootScope.$emit('common.showerror', '苹果数量太多了');
        }else{
            fruitsService.set_apple_count(newVal);
        }
    }, true);
    $scope.$watch('orange_count', function (newVal, oldVal, $scope) {
        if (newVal > 10){
            $rootScope.$emit('common.showerror', '桔子数量太多了');
        }else{
            fruitsService.set_orange_count(newVal);
        }
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
