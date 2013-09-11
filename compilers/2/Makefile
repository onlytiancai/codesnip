app: lex.yy.c name.y.c name.y.h
	gcc -o app lex.yy.c name.y.c 

lex.yy.c: name.y.h
	lex name.lex

name.y.c name.y.h: name.y
	yacc -d -o name.y.c name.y

test:
	./app <names.txt
