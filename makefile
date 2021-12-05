nnUtils.o:
	nvcc -Xcompiler -rdynamic -g -G -c nnUtils.cu

test.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -g -G -c test.cu

test: nnUtils.o test.o
	nvcc -Xcompiler -rdynamic -g -G -o test nnUtils.o test.o  -lm

clean:
	rm -rf *.o test
