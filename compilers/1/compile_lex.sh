lex hello.lex
gcc lex.yy.c -o lex.o 
./lex.o < hello.lex
