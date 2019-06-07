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


QUnit.test('+-', function(assert) {
    var input = 'print 1 + 2 + 3 - 2';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([4], printData);
});

QUnit.test('+-*/', function(assert) {
    var input = 'print 1 + 2 * 3';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([7], printData);
});

QUnit.test('(+-)*/', function(assert) {
    var input = 'print (1 + 2) * 3';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([9], printData);
});

QUnit.test('>,<,>=,<=,==,!=', function(assert) {
    var input = 'print 1 < 0\n' +
        'print 1 < 2\n' +
        'print 1 <= 1\n' +
        'print 1 <= 2\n' +
        'print 1 <= 0\n' +
        'print 1 > 0\n' +
        'print 1 > 2\n' +
        'print 1 >= 1\n' +
        'print 1 >= 2\n' +
        'print 1 >= 0\n' +
        'print 1 == 1\n' +
        'print 1 == 2\n' +
        'print 1 != 1\n' +
        'print 1 != 2\n';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([false, true, true, true, false, true, false, true, false, true, true, false, false, true], printData);
});

QUnit.test('and', function(assert) {
    var input = 'print 2 > 1 and 3 > 2';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([true], printData);
});

QUnit.test('> and >', function(assert) {
    var input = 'print 2 > 1 + 1 and 3 > 2';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([false], printData);
});

QUnit.test('> + and > or >', function(assert) {
    var input = 'print 2 > 1 + 1 and 3 > 2 or 1 > 0';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([true], printData);
});

QUnit.test('> or > + and >', function(assert) {
    var input = 'print 1 > 0 or 2 > 1 + 1 and 3 > 2';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([true], printData);
});

QUnit.test("assign", function(assert) {
    var input = 'a = 1 + 2\n' +
        '    print a\n';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([3], printData);
});

QUnit.test("if", function(assert) {
    var input = 'if 1 > 2 then\n' +
        '    print 1\n' +
        'end\n' +
        'if 2 > 1 then\n' +
        '    print 2\n' +
        'end\n';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([2], printData);
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
    console.log(ast);

    ast.eval();
    assert.deepEqual([2, 4, 6, 8], printData);
});

QUnit.test("def foo", function(assert) {
    var input = 'def foo()\n' +
        '  print 1 + 1\n' +
        'end\n' +
        'foo()\n';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([2], printData);
});


QUnit.test("def add", function(assert) {
    var input = 'def foo(a, b)\n' +
        '  print a + b\n' +
        'end\n' +
        'foo(1, 2)\n';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([3], printData);
});


QUnit.test("def max", function(assert) {
    var input = 'def foo(a, b)\n' +
        '  if a > b then\n' +
        '    print a\n' +
        '  end\n' +
        '  if b > a then\n' +
        '    print b\n' +
        '  end\n' +        
        'end\n' +
        'foo(1 + 2, 2 + 2)\n';
    var ast = parser.parse(input);
    console.log(ast);
    assert.equal(ast.body.type, 'SeqStat');
    assert.equal(ast.body.body[0].type, 'def');
    assert.equal(ast.body.body[1].type, 'call');

    ast.eval();
    assert.deepEqual([4], printData);
});


QUnit.test("call exp", function(assert) {
    var input = 'def add(a, b)\n' +
        '  return a + b\n' +
        'end\n' +
        'print add(1, add(1, 2))\n';
    var ast = parser.parse(input);
    console.log(ast);

    ast.eval();
    assert.deepEqual([4], printData);
});