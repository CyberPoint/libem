EM_DIR=.

#
# you must set the next three variables to the appropriate values for your environment
# LAPACK_INC_PATH must be the path to lapacke.h
# LAPACK_PATH must be the path to libpacke.a and liblapack.a
# BLAS_PATH must be the path to librefblas.a
#
LAPACK_INC_PATH=/home/julrich/code/vulcan/src/lapack/lapacke/include
LAPACK_PATH=/home/julrich/code/vulcan/build/lib
BLAS_PATH=/home/julrich/code/vulcan/build/lib

$(CXX) = g++ -g3 -gdwarf2
$(CC) = gcc -g3 -gdwarf2

all: em_algorithm

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: em_algorithm

#///// LINK STEPS /////


OBJS = Matrix.o EM_Algorithm.o  sample_main.o


#-- Build-only target --
em_algorithm: $(OBJS)
	@echo LAPACK_PATH is set to $(LAPACK_PATH)
	@echo BLAS_PATH is set to $(BLAS_PATH)
	$(CXX) -o em_algorithm $(OBJS) -L$(LAPACK_PATH) -llapacke -llapack -L$(BLAS_PATH) -lrefblas -lgfortran 

#///// COMPILE STEPS /////


Matrix.o: $(EM_DIR)/Matrix.cc $(EM_DIR)/Matrix.h
	@echo LAPACK_INC_PATH is set to $(LAPACK_INC_PATH)
	$(CXX) -c $(EM_DIR)/Matrix.cc -I$(LAPACK_INC_PATH) 

EM_Algorithm.o: $(EM_DIR)/EM_Algorithm.cpp $(EM_DIR)/EM_Algorithm.h
	$(CXX) -c $(EM_DIR)/EM_Algorithm.cpp 

sample_main.o: $(EM_DIR)/sample_main.cpp $(EM_DIR)/EM_Algorithm.h $(EM_DIR)/Matrix.h
	$(CXX) -c $(EM_DIR)/sample_main.cpp -I $(EM_DIR)
		

#///// clean-up /////
clean:
	rm -f *.o em_algorithm
