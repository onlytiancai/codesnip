INC=-I /usr/local/include/CUnit
LIB=-L /usr/local/lib/

.PHONY: test 
test: testcase.c
	gcc -o test $(INC) $(LIB) -g  $^ -l cunit

helloworld.o: helloworld.c
	gcc helloworld.c -o helloworld.o

.PHONY: lint 
lint:
	splint helloworld.c -temptrans -mustfreefresh -usedef

.PHONY: run
run:
	./helloworld.o

.PHONY: clean
clean:
	rm *.o 
