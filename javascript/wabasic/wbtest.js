var grammar = $.ajax({ type: "GET", url: './wb.pegjs', async: false }).responseText;
var parser = peg.generate(grammar);

// 临时存储打印的数据
var printData = [];
// 拦截 println 函数，捕获打印的数据
function println(d) {
    printData.push(d);
    console.log(d);
}

QUnit.module("Parse Test", {
    beforeEach: function() { printData = []; },
});

QUnit.test('print', function(assert) {
    var input = 'print 1 + 2';
    var ast = parser.parse(input);
    console.log(ast);

    ast.body.eval();
    assert.deepEqual([3], printData);
});

QUnit.test("while", function(assert) {
    var input = 'a = 1\n' +
        'while a < 10 then\n' +
        '  if a % 2 == 0 then\n' +
        '    print a\n' +
        '  end\n' +
        '  a = a + 1\n' +
        'end\n';
    var ast = parser.parse(input);
    ast.body.eval();
    assert.deepEqual([2, 4, 6, 8], printData);
});