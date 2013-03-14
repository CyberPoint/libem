#*********************************************************************************
# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
#
# This software is offered under the NewBSD license:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the CyberPoint International, LLC nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*********************************************************************************/

EM_DIR=.

# specify paths to BLAS and LAPACK includes here
include gaussmix.inc

CXX = g++ -g3 -gdwarf-2 -fopenmp -fPIC
CC = gcc -g3 -gdwarf-2 -fopenmp -fPIC

all: gaussmix

debug: CXX = g++ -g3 -gdwarf-2 -fopenmp -DDEBUG -g -fPIC
debug: CC = gcc -g3 -gdwarf-2 -fopenmp -DDEBUG -g -fPIC
debug: gaussmix    

#///// LINK STEPS /////


OBJS = Matrix.o Adapt.o KMeans.o GaussMix.o  

#-- Build-only target --
gaussmix: $(OBJS) sample_main.o
	@echo LAPACK_PATH is set to $(LAPACK_PATH)
	@echo BLAS_PATH is set to $(BLAS_PATH)
	ar -cvq libgaussmix.a $(OBJS)
	gcc -shared -Wl,-soname,libgaussmix.so.1 -o libgaussmix.so.1.0 $(OBJS)
	ln -fs libgaussmix.so.1.0 libgaussmix.so.1
	ln -fs libgaussmix.so.1.0 libgaussmix.so
	$(CXX) -o gaussmix sample_main.o -L$(EM_DIR) -lgaussmix -L$(LAPACK_PATH) -llapacke -llapack \
		-L$(BLAS_PATH) -l$(BLAS_NAME) -lgfortran 

#///// COMPILE STEPS /////


Matrix.o: $(EM_DIR)/Matrix.cpp $(EM_DIR)/Matrix.h
	@echo LAPACK_INC_PATH is set to $(LAPACK_INC_PATH)
	$(CXX) -c $(EM_DIR)/Matrix.cpp -I$(LAPACK_INC_PATH) 

KMeans.o: $(EM_DIR)/KMeans.cpp $(EM_DIR)/KMeans.h
	$(CXX) -c $(EM_DIR)/KMeans.cpp 

Adapt.o: $(EM_DIR)/Adapt.cpp $(EM_DIR)/Adapt.h
	$(CXX) -c $(EM_DIR)/Adapt.cpp -I$(LAPACK_INC_PATH)

GaussMix.o: $(EM_DIR)/GaussMix.cpp $(EM_DIR)/GaussMix.h
	$(CXX) -c $(EM_DIR)/GaussMix.cpp -I$(LAPACK_INC_PATH) 

sample_main.o: $(EM_DIR)/sample_main.cpp $(EM_DIR)/GaussMix.h $(EM_DIR)/Matrix.h
	$(CXX) -c $(EM_DIR)/sample_main.cpp -I $(EM_DIR) 


#///// clean-up /////
clean:
	rm -f *.o gaussmix libgaussmix.*
