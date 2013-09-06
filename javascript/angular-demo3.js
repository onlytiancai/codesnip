var myApp = angular.module('myApp', []);
myApp.directive('loading', function(){
    return {
        restrict: "A",
        controller: function ($scope, $element, $attrs) {
            var css =  {'opacity':'0.6','background-color':'#777'};
            var imgsrc = $attrs.loadingimg || 'aj-blue.gif';

            $scope.$watch('isloading', function (newVal, oldVal, $scope) {
                console.log('watch:isloading', newVal);
                if(newVal){
                    var offset = $element.offset();
                    var height = $element.height();
                    var width = $element.width();
                    var id = $element[0].id;
                    var newid = id + "waiting";

                    $element[0].waitid = newid;

                    var div = $("<div />").css({'position':'absolute','top':offset.top,'left':offset.left,'height':height,'width':width}).css(css)
                    .attr('id',newid).appendTo("body");

                    var img = $("<img />").attr("src", imgsrc);
                    var imgheight = img.height();
                    var imgwidth = img.width();
                    img.css({'position':'absolute','top':((height/2)-(imgheight/2))/*offset.top +(height/2)-(imgheight/2)*/,'left':((width/2)-(imgwidth/2)),'z-index':3000});
                    img.appendTo(div);
                }else{
                    var waitid = $element[0].waitid;
                    $("#"+waitid).remove();
                }
            }, true);
        }
    }
})
.controller('myCtrl', function($scope, $timeout) {
    console.log('$scope.isloading = true;')
    $scope.isloading= true;
    $timeout(function(){
        console.log('$scope.isloading = false;')
        $scope.isloading = false; 
    }, 3000);
});
