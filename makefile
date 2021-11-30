nnUtils.o:
	gcc -g -c nnUtils.c

test.o: nnUtils.o
	gcc -g -c test.c

test: nnUtils.o test.o
	gcc -g -o test nnUtils.o test.o  -lm

clean:
	rm -rf *.o test
