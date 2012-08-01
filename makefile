VULCAN_PATH=../Vulcan/vulcan
VULCAN_INC=$(VULCAN_PATH)/include
VULCAN_SRC=$(VULCAN_PATH)/src
EM_DIR=.

LAPACK_PATH=/opt/lapack-3.4.1
#LAPACK_PATH=/home/egarbee/lapack-3.4.1
#LAPACK_PATH=/tmp/lapack-3.4.1

BLAS_PATH=/opt/BLAS
#BLAS_PATH=/tmp/lapack-3.4.1
#BLAS_PATH=/usr/lib64

$(CXX) = g++ -g3 -gdwarf2
$(CC) = gcc -g3 -gdwarf2

all: em_algorithm

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: em_algorithm

#///// LINK STEPS /////


OBJS = Matrix.o EM_Algorithm.o


#-- Build-only target --
dummy: $(OBJS)

em_algorithm: $(OBJS)
	$(CXX) -o em_algorithm $(OBJS) -L$(LAPACK_PATH) -llapacke -llapack -L$(BLAS_PATH) -lblas -lgfortran 

#///// COMPILE STEPS /////

#-- utils --

Matrix.o: $(VULCAN_SRC)/utils/Matrix.cc $(VULCAN_INC)/utils/Matrix.h
	$(CXX) -c $(VULCAN_SRC)/utils/Matrix.cc -I$(VULCAN_INC)/utils -I/usr/local/include \
	-I$(LAPACK_PATH)/lapacke/include #$(LAPACK_PATH)/liblapack.a $(LAPACK_PATH)/liblapacke.a $(BLAS_PATH)/libblas.a

EM_Algorithm.o: $(EM_DIR)/EM_Algorithm.cpp $(EM_DIR)/EM_Algorithm.h
	$(CXX) -c $(EM_DIR)/EM_Algorithm.cpp 
		

#///// clean-up /////
clean:
	rm -f *.o em_algorithm
