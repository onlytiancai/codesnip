@builtin "whitespace.ne"
@builtin "string.ne"


@{% var g = require('./postprocessors'); %}


final    -> null                     {% g.finalEmpty %}
          | stmtList                 {% g.final %}
stmts    -> "{" _ (stmtList _):? "}" {% g.stmtsBlock %}
          | stmt                     {% g.stmtsSingle %}
stmtList -> stmt                     {% g.stmtListSingle %}
          | stmtList __ stmt         {% g.stmtList %}


stmt  -> expr    {% id %}
       | comment {% id %}
       | if      {% id %}
       | while   {% id %}
       | assign  {% id %}
       | return  {% id %}
       | class   {% id %}
       | func    {% id %}

expr    -> id         {% g.expr %}
         | str        {% g.expr %}
         | int        {% g.expr %}
         | float      {% g.expr %}
         | bool       {% g.expr %}
         | nul        {% g.expr %}
         | methodCall {% g.expr %}
         | closure    {% g.expr %}
         | funcCall   {% g.expr %}
comment -> "#" [^\n]:*                             {% g.comment %}
if      -> "if" __ expr __ stmts elseif:* else:?   {% g.if_ %}
while   -> "while" __ expr __ stmts                {% g.while_ %}
assign  -> id _ "=" _ expr                         {% g.assign %}
return  -> "return" __ expr                        {% g.return_ %}
class   -> "class" __ id (":" _ id):? _ methodList {% g.class_ %}
func    -> "func" __ id argDefList stmts           {% g.func %}


id         -> [a-zA-Z] [a-zA-Z0-9_]:* {% g.id %}
str        -> dqstring                {% g.str %}
            | sqstring                {% g.str %}
int        -> [0-9]:+                 {% g.int %}
float      -> int:? "." int           {% g.float %}
bool       -> "true"                  {% g.boolTrue %}
            | "false"                 {% g.boolFalse %}
nul        -> "null"                  {% g.null_ %}
methodCall -> id "." id argCallList   {% g.methodCall %}
closure    -> "func" argDefList stmts {% g.closure %}
funcCall   -> id argCallList          {% g.funcCall %}




# HELPERS

argCallList -> "(" (expr ("," _ expr):*):? ")" {% g.argList %}
argDefList  -> "(" (id ("," _ id):*):? ")"     {% g.argList %}
methodList  -> "{" _ (method _):* "}"          {% g.methodList %}
method      -> id argDefList _ stmts           {% g.method %}
elseif      -> _ "elseif" __ expr __ stmts     {% g.elseif %}
else        -> _ "else" __ stmts               {% g.else_ %}