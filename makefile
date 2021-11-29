nnUtils.o:
	gcc -c nnUtils.c

test.o: nnUtils.o
	gcc -c test.c 

test: nnUtils.o test.o
	gcc -g -o test nnUtils.o test.o

clean:
	rm -rf *.o test
