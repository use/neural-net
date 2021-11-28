nnUtils.o:
	nvcc -g -c nnUtils.cu

test.o: nnUtils.o
	nvcc -g -c test.cu nnUtils.o

test: nnUtils.o test.o
	nvcc -g -o test nnUtils.o test.o

clean:
	rm -rf *.o test
