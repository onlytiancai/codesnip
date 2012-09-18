var wawatests = [];

(function(){
    var test1 = {
        url : 'betest.html',
        timeout: 10,
        action: function(page){
            page.find('button').click();
        },
        callback: function(page, assert){
            var p_text = page.find('p').text();
            assert.assertEqual('first test', p_text, 'haha'); 
        }
    };
    wawatests.push(test1);
})();
