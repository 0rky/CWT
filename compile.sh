gcc -fopenmp -ggdb `pkg-config --cflags opencv` -o `basename conv.c .c` cwt.c filter.c normfilter.c `pkg-config --libs opencv` -I/home/manas/fftw-3.3.4/include -lfftw3f_omp -lfftw3f -lm 
