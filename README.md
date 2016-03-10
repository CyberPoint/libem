libem
=====

An Adapted Gaussian Mixture Model library
Developed by David Ritch, James Ulrich, Riva Borbely, Mark Raugas, and Elizabeth Garbee, and Mike West at
Cyberpoint LLC (www.cyberpointllc.com).

Copyright 2013 CyberPoint International LLC.

The contents of libGaussMix are offered under the NewBSD license:

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: (1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. (2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. (3) Neither the name of the CyberPoint International, LLC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Dependencies:

This library has been tested on 64-bit Ubuntu 15.10.  Please install the following dependencies:

cmake libatlas-dev libblas-dev gfortran liblapack-dev liblapacke-dev

## Building libem:

LibEM OpenCL support is experimental.  These instructions assume OpenCL and CUDA are installed:

$mkdir -p build
$cd ./build
$LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/nvidia:/usr/lib/nvidia
$cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH:string="/usr/local/lapack" -DCMAKE_CXX_FLAGS:string="-I/usr/include/lapacke -I/usr/local/lapack/include -L/usr/local/lapack/lib" -DCMAKE_MODULE_PATH:string="`pwd`/../cmake/Modules" ..
$make all
$cp ../oclEstep.cl .


