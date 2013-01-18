0: main main2 
	echo 'make finished'
	
main: main.c ./libuv/libuv.a
	gcc -o run main.c ./libuv/libuv.a -lpthread -lrt

main2: main2.c ./libuv/libuv.a
	gcc -o run2 main2.c ./libuv/libuv.a -lpthread -lrt

clean: 
	rm run* -f
