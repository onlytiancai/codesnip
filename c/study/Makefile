INC=-I /usr/local/include/CUnit
LIB=-L /usr/local/lib/

test: testcase.c
	gcc -o test.o $(INC) $(LIB) -g  $^ -l cunit
	./test.o

helloworld.o: helloworld.c
	gcc helloworld.c -o helloworld.o

lint:
	splint helloworld.c -temptrans -mustfreefresh -usedef

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

.PHONY: test lint run clean prof
