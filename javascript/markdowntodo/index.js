var markdown = require( "markdown" ).markdown;

exports.tohtml = function(text)
{
    return markdown.toHTML( text );
}