EM_DIR=.

#
# you must set the next three variables to the appropriate values for your environment
# LAPACK_INC_PATH must be the path to lapacke.h
# LAPACK_PATH must be the path to libpacke.a and liblapack.a
# BLAS_PATH must be the path to librefblas.a
#
LAPACK_INC_PATH=
LAPACK_PATH=
BLAS_PATH=

CXX = g++ -g3 -gdwarf-2 -fopenmp
CC = gcc -g3 -gdwarf-2 -fopenmp

all: gaussmix

debug: CXX = g++ -g3 -gdwarf-2 -fopenmp -DDEBUG -g
debug: CC = gcc -g3 -gdwarf-2 -fopenmp -DDEBUG -g
debug: gaussmix    

#///// LINK STEPS /////


OBJS = Matrix.o Adapt.o KMeans.o GaussMix.o sample_main.o


#-- Build-only target --
gaussmix: $(OBJS)
	@echo LAPACK_PATH is set to $(LAPACK_PATH)
	@echo BLAS_PATH is set to $(BLAS_PATH)
	$(CXX) -o gaussmix $(OBJS) -L$(LAPACK_PATH) -llapacke -llapack -L$(BLAS_PATH) -lrefblas -lgfortran 

#///// COMPILE STEPS /////


Matrix.o: $(EM_DIR)/Matrix.cpp $(EM_DIR)/Matrix.h
	@echo LAPACK_INC_PATH is set to $(LAPACK_INC_PATH)
	$(CXX) -c $(EM_DIR)/Matrix.cpp -I$(LAPACK_INC_PATH) 

KMeans.o: $(EM_DIR)/KMeans.cpp $(EM_DIR)/KMeans.h
	$(CXX) -c $(EM_DIR)/KMeans.cpp 

Adapt.o: $(EM_DIR)/Adapt.cpp $(EM_DIR)/Adapt.h
	$(CXX) -c $(EM_DIR)/Adapt.cpp 

GaussMix.o: $(EM_DIR)/GaussMix.cpp $(EM_DIR)/GaussMix.h
	$(CXX) -c $(EM_DIR)/GaussMix.cpp 

sample_main.o: $(EM_DIR)/sample_main.cpp $(EM_DIR)/GaussMix.h $(EM_DIR)/Matrix.h
	$(CXX) -c $(EM_DIR)/sample_main.cpp -I $(EM_DIR)
		

#///// clean-up /////
clean:
	rm -f *.o gaussmix
