helloworld.o: helloworld.c
	splint helloworld.c -temptrans -mustfreefresh -usedef
	gcc helloworld.c -o helloworld.o

run:
	./helloworld.o
