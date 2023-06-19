var md = require( "markdown" ).markdown;

exports.tohtml = function(text)
{
    return md.toHTML( text );
}

const text = `
### ttt

- todo: aaa
- doing: bbb
- done: ccc
`;

const mdTree = md.parse( text );
console.log(mdTree);

const htmlTree = md.toHTMLTree(mdTree)
console.log(htmlTree);

const html = md.renderJsonML(htmlTree);
console.log( html );