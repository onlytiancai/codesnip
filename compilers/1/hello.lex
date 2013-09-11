%{
int wordCount = 0;
%}
chars [A-za-z\_\'\.\"]
numbers ([0-9])+
delim [" "\n\t]
whitespace {delim}+
words {chars}+
%%

{words} { wordCount++; /* increase the word count by one*/ }
{whitespace} { /* do nothing*/ }
{numbers} { /* one may want to add some processing here*/ }
%%

int main() {
    yylex(); /* start the analysis*/
    printf(" No of words: %d\n", wordCount);
    return 0;
}
int yywrap()
{
    return 1;
}
