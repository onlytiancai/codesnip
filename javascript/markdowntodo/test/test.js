tohtml = require('../index').tohtml
var assert  = require("assert");


it("test", function()
{
    var ret = tohtml("Hello *World*!");
    assert.equal(ret, '<p>Hello <em>World</em>!</p>');
});
