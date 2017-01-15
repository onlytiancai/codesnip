all: tcpserver.o echo.o selectserver.o epollserver.o test.o
tcpserver.o: tcpserver.c 
	gcc -o $@ $^ 

echo.o: echo.c
	gcc -o $@ $^

selectserver.o: selectserver.c
	gcc -o $@ $^ -g

epollserver.o: epollserver.c
	gcc -o $@ $^ -g

test.o: test.c
	gcc -o $@ $^ -g
