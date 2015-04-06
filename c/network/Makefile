all: tcpserver.o echo.o selectserver.o
tcpserver.o: tcpserver.c 
	gcc -o $@ $^ 

echo.o: echo.c
	gcc -o $@ $^

selectserver.o: selectserver.c
	gcc -o $@ $^ -g
