%{
#typedef char* string; /* to specify token types as char* */
#define YYSTYPE string /* a Yacc variable which has the value of returned token */
%}
%token NAME EQ AGE
%%

file : record file
    | record
    ;

record: NAME EQ AGE {
    printf("%s is now %s years old!!!", $1, $3);}
    ;

%%

int yyerror(char* msg)
{
    printf("Error: %s encountered at line number:%d\n", msg, yylineno);
}

void main()
{
    yyparse();
}
