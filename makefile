project: nnUtils.o project.o
	nvcc -Xcompiler -rdynamic -g -G -o project nnUtils.o project.o  -lm

test: nnUtils.o test.o
	nvcc -Xcompiler -rdynamic -g -G -o test nnUtils.o test.o  -lm

nnUtils.o:
	nvcc -Xcompiler -rdynamic -g -G -c nnUtils.cu

project.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -g -G -c project.cu

test.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -g -G -c test.cu

all: project test

clean:
	rm -rf *.o project test
