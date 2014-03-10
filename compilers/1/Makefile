lex.o: lex.yy.c
	gcc lex.yy.c -o lex.o 
lex.yy.c: hello.lex
	lex hello.lex
test:
	./lex.o < hello.lex
