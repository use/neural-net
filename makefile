project: nnUtils.o project.o
	nvcc -Xcompiler -rdynamic -lineinfo -g -o project nnUtils.o project.o  -lm -lcudadevrt -rdc=true
#	nvcc -Xptxas -O3,-v -o project nnUtils.o project.o  -lm -lcudadevrt -rdc=true

test: nnUtils.o test.o
	nvcc -Xcompiler -rdynamic -lineinfo -g -o test nnUtils.o test.o  -lm -lcudadevrt -rdc=true
#	nvcc -Xptxas -O3,-v -o test nnUtils.o test.o  -lm -lcudadevrt -rdc=true

nnUtils.o:
	nvcc -Xcompiler -rdynamic -lineinfo -g -c nnUtils.cu -lcudadevrt -rdc=true
#	nvcc -Xptxas -O3,-v -c nnUtils.cu -lcudadevrt -rdc=true

project.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -lineinfo -g -c project.cu -lcudadevrt -rdc=true
#	nvcc -Xptxas -O3,-v -c project.cu -lcudadevrt -rdc=true

test.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -lineinfo -g -c test.cu -lcudadevrt -rdc=true
#	nvcc -Xptxas -O3,-v -c test.cu -lcudadevrt -rdc=true

all: project test

clean:
	rm -rf *.o project test
