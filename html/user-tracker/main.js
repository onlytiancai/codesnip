if (!String.format) {
  String.format = function(format) {
    var args = Array.prototype.slice.call(arguments, 1);
    return format.replace(/{(\d+)}/g, function(match, number) { 
      return typeof args[number] != 'undefined'
        ? args[number] 
        : match
      ;
    });
  };
}

function reportEvent(type, text) {
    var msg = String.format('{0}:({1}) clicked.', type, text);
    console.log(msg);
    document.getElementById("msg").innerText = msg;
}

document.addEventListener('DOMContentLoaded', function () {
    console.log('init');
    document.addEventListener('click', function(e){
        var ele = e.target; 
        if (ele.nodeType != 1)  return; // Element
        switch(ele.tagName) {
            case "BUTTON":
                reportEvent('button', ele.innerText);
                break;
            case "INPUT":
                if (ele.type == 'button') {
                    reportEvent('button', ele.value);
                }
                break;
            case "A":
                reportEvent('link', ele.text + ' ' + ele.href);
                break;
            default:
                reportEvent(ele.nodeName, ele.innerText);
                break;
        }
    });
});
