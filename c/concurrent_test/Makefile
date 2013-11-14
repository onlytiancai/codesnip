concurrent_test.o: concurrent_test.c
	gcc -Wall -g -pg -o concurrent_test.o concurrent_test.c -lpthread
test:
	./concurrent_test.o test
profiler:
	gprof -b concurrent_test.o gmon.out
