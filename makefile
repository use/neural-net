project: nnUtils.o project.o
	nvcc -Xcompiler -rdynamic -g -G -o project nnUtils.o project.o  -lm -lcudadevrt -rdc=true

test: nnUtils.o test.o
	nvcc -Xcompiler -rdynamic -g -G -o test nnUtils.o test.o  -lm -lcudadevrt -rdc=true

nnUtils.o:
	nvcc -Xcompiler -rdynamic -g -G -c nnUtils.cu -lcudadevrt -rdc=true

project.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -g -G -c project.cu -lcudadevrt -rdc=true

test.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -g -G -c test.cu -lcudadevrt -rdc=true

all: project test

clean:
	rm -rf *.o project test
