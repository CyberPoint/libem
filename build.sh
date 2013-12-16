#!/bin/sh

rm -rf ./build && mkdir -p ./build 
rm -rf ./cmake/findopencl && git submodule update --init ./cmake/findopencl
pushd ./build

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/nvidia:/usr/lib/nvidia

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH:string="/usr/local/lapack" -DCMAKE_CXX_FLAGS:string="-I/usr/local/lapack/include -L/usr/local/lapack/lib" -DCMAKE_MODULE_PATH:string="`pwd`/../cmake/Modules" ..

make all
popd
