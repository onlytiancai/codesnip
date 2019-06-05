QUnit.module("utils group");
QUnit.test("Set#add", function(assert) {
    var Set = utils.Set;
    var set = new Set();
    set.add(1);
    set.add(2);
    set.add(1);
    assert.deepEqual(set.list, [1, 2], 'Passed')

    set = new Set([1, 2]);
    set.add(1).add(2).add(3);
    assert.deepEqual(set.list, [1, 2, 3], 'Passed')
});

QUnit.test("Set#contains", function(assert) {
    var Set = utils.Set;
    var set = new Set([1, 2]);
    assert.ok(set.contains(1), 'Passed')
    assert.notOk(set.contains(3), 'Passed')
});

QUnit.test("Set#intersect", function(assert) {
    var Set = utils.Set;
    assert.ok(new Set([1, 2]).intersect(new Set([2, 3])).length > 0, 'Passed')
    assert.ok(new Set([1, 2]).intersect(new Set([3, 4])) == 0, 'Passed')
});


QUnit.module("lexer group");

QUnit.test("Num", function(assert) {
    var Num = lexer.Num;
    assert.ok(new Num('123').parse().accepted());
    assert.notOk(new Num('a').parse().accepted());
    assert.notOk(new Num('123a').parse().accepted());
});

QUnit.test("1 + 2", function(assert) {

    var tokens = lexer.parse('1 + 2');
    assert.deepEqual(tokens, ['1', '+', '2'], 'Passed')
});

QUnit.test("1+2", function(assert) {

    var tokens = lexer.parse('1+2');
    assert.deepEqual(tokens, ['1', '+', '2'], 'Passed')
});

QUnit.module("parser group");
QUnit.test("parser test", function(assert) {
    assert.ok(1 == "1", "Passed!");
});