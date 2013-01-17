main.o: main.c ./http_parser/http_parser.o
	gcc main.c ./http_parser/http_parser.o -o main.o

./http_parser/http_parser.o: ./http_parser/http_parser.c ./http_parser/http_parser.h
	gcc -c ./http_parser/http_parser.c -o ./http_parser/http_parser.o
