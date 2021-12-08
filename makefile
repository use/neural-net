project: nnUtils.o test.o
	nvcc -Xcompiler -rdynamic -g -G -o project nnUtils.o test.o  -lm

nnUtils.o:
	nvcc -Xcompiler -rdynamic -g -G -c nnUtils.cu

test.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -g -G -c test.cu

clean:
	rm -rf *.o project
