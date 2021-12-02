nnUtils.o:
	nvcc -g -c nnUtils.cu

test.o: nnUtils.o
	nvcc -g -c test.cu

test: nnUtils.o test.o
	nvcc -g -o test nnUtils.o test.o  -lm

clean:
	rm -rf *.o test
