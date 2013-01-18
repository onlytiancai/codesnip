0: main main2 main3 main4
	echo 'make finished'
	
main: main.c ./libuv/libuv.a
	gcc -o run main.c ./libuv/libuv.a -lpthread -lrt

main2: main2.c ./libuv/libuv.a
	gcc -o run2 main2.c ./libuv/libuv.a -lpthread -lrt

main3: main3.c ./libuv/libuv.a
	gcc -o run3 main3.c ./libuv/libuv.a -lpthread -lrt

main4: main4.c ./http_parser/http_parser.o
	gcc -o run4 main4.c ./http_parser/http_parser.o 

./http_parser/http_parser.o: ./http_parser/http_parser.c ./http_parser/http_parser.h
	gcc -c ./http_parser/http_parser.c -o ./http_parser/http_parser.o

clean: 
	rm run* -f
