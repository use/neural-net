project: nnUtils.o project.o
	nvcc -Xcompiler -rdynamic -g -G -o project nnUtils.o project.o  -lm

nnUtils.o:
	nvcc -Xcompiler -rdynamic -g -G -c nnUtils.cu

project.o: nnUtils.o
	nvcc -Xcompiler -rdynamic -g -G -c project.cu

clean:
	rm -rf *.o project
