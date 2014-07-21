INC=-I /usr/local/include/CUnit
LIB=-L /usr/local/lib/

all: socket_test.o helloworld.o array_list.o

array_list.o: array_list.c
	gcc -g -o array_list.o array_list.c

socket_test.o: socket_test.c
	gcc -g -o socket_test.o socket_test.c

helloworld.o: helloworld.c
	gcc -g -o helloworld.o helloworld.c

lint:
	splint helloworld.c -temptrans -mustfreefresh -usedef
	splint socket_test.c -warnposix -usedef

test: testcase.c
	gcc -o test.o $(INC) $(LIB) -g  $^ -l cunit
	./test.o

run:
	./helloworld.o

prof:
	rm *.o -f
	rm gmon.out -f 
	gcc -o helloworld.o -pg -g helloworld.c
	./helloworld.o
	gprof -b -A -p -q helloworld.o gmon.out >prof_info.txt 

	

clean:
	rm *.o -f
	rm gmon.out -f
	rm prof_info.txt -f

.PHONY: all test lint run clean prof
