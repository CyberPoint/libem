/*********************************************************************************
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
**********************************************************************************/

/*! \file GaussMix.cpp
    \version 1.0
    \brief core libGaussMix++ em algorithm method implementations.
    \author Elizabeth Garbee & CyberPoint Labs, CyberPoint International LLC
    \date Summer 2012

*/

/*! \mainpage libGaussMix: An Expectation Maximization Algorithm for Training Gaussian Mixture Models
*
*\authors Elizabeth Garbee & CyberPoint Labs
*\date February 7, 2013
*
* Copyright 2013 CyberPoint International LLC.
*
* The contents of libGaussMix are offered under the NewBSD license:
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     (1) Redistributions of source code must retain the above copyright
*         notice, this list of conditions and the following disclaimer.
*     (2) Redistributions in binary form must reproduce the above copyright
*         notice, this list of conditions and the following disclaimer in the
*         documentation and/or other materials provided with the distribution.
*     (3) Neither the name of the CyberPoint International, LLC nor the
*         names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*\section intro_sec Introduction
*
* A Gaussian mixture model is method of approximating, as a linear combination of multivariate Gaussian density functions, an unknown density function from which a given
* data set is presumed to be drawn. The means and covariance matrices of these Gaussian densities are sometimes referred to as the "hidden" parameters of the data set.
* 
* Expectation Maximization (EM) is perhaps the most popular technique for discovering the parameters of a mixture with a given number of components. Here's the usual scenario:
* one is given a set of N data points in an M-dimensional space. One seeks to fit to the data a set of k multivariate Gaussian distributions (or "clusters") that best represent the observed distribution of the data
* points, where k is specified, but the means and covariances of the Gaussians are unknown. The desired output includes the means and covariances of the Gaussians, along with the cluster weights
* (coefficients of the linear combination of Gaussians). One may think of the data as being drawn by first randomly choosing a particular Gaussian according to the cluster weights, and then
* choosing a data point according to the selected distribution. The overall probability density assigned to a point is then given by the sum over k of P(n|k)P(k) where k denotes the cluster, P(k) its probability, and P(n|k) denotes
* the probability density of the point given the cluster. The likelihood density L of the data is the product of these sums, taken over all data points (which are assumed to be independent and identically distributed).
* The EM algorithm produces an estimate of the P(k) and the P(n|k),by attempting to maximize L. Specifically, it yields the estimated means mu(k), covariance matrices sigma(k), and weights P(k) for each Gaussian.
*
* An outline of the EM calculations is best described by working backwards from L. Since the data points are (assumed) to be independent, L is the product of the probability densities of
* each observed data point, which splits into a contribution P(n|k) from each Gaussian (these contributions are sometimes called the data point "mixture weights").
* In the language of EM, the L and P(n|k) calculations, given known mu(k), sigma(k), and P(k), comprise the algorithm's "expectation step", or E step.
* Now if the P(n|k)'s are known, we can derive from them maximum likelihood estimates of the mu(k), sigma(k), and P(k), in a process called the "maximization step", or M step, from which we then obtain a new set of P(n|k), and a new (better) L.
*
* The power of the EM algorithm comes from a theorem (see the first reference) stating that, starting from any reasonable guess of parameter values, an iteration of an E step followed by an M
* step will always increase the likelihood value L, and that repeated iterations between these two steps will converge to (at least) a local maximum. Often, the convergence is indeed to the
* global maximum. The EM algorithm, in brief, goes like this: 
* - Guess starting values for the mu(k), sigma(k) and P(k) for each Gaussian,
* - Repeat: an E step to get a new L and new P(n|k)s, followed by an M step to get new mu(k)s, sigma(k)s, and P(k)s,
* - Stop when the value of L is no longer meaningfully changing.
*
* The libGaussMix implementation of the EM algorithm uses the "KMeans" clustering algorithm to provide the initial guesses for the means of the K Gaussians, in order to increase the efficiency and efficacy of
* EM. 
*
* The libGaussMix code base also includes support for adapation of a GMM to a specific subpopulation of the population on which it was trained. In a sense, this
* "biases" the model towards the subpopulation. If the population splits into distinct subpopulations, one may then classify a sample as belonging to a particular
* subpopulation, by assiging it to the subpopulation  under whose adapted GMM it has the highest likelihood density. One often normalizes
* these scores by the unadapted GMM density. See reference 2 for details. This functionality is provided by the adapt() method of the GaussMix API.
*
* \section usage_sec Usage
*
* The libGaussMix library builds into both .a and .so files. The API is provided in GaussMix.h; API implementations are contained in GaussMix.cpp. The implementations
* use helper routines provided in Matrix.h/ Matrix.cpp, which in turn wrap LAPACK linear algebra functions, and in KMeans.h/ KMeans.cpp, to generate initial model parameter guesses.
* To build the lib/API into an executable using the sample driver function provided in sample_main.cpp, follow these steps:
*
* - 1. Install BLAS, LAPACK, and LAPACKE on your machine if they're not already there (c.f. http://www.netlib.org/lapack/, which bundles a reference vesion of BLAS).
* - 2. Update the environment variables in libgaussmix.inc to point to your environment's BLAS and LAPACK header and library locations.
* - 3. run make on the libGaussMix makefile.
* - 4. run the resulting executable via: "gaussmix <data_file> <num_dimensions> <num_data_points> <num_clusters>". \n
*      Try using one of the sample *.csv or *.svm data files as the first argument: e.g. "gaussmix multid_data_1.csv 3 20 2".
*
* The libGaussMix source code has been built and tested on SL6.2 using gcc 4.4.5.
* 
*
* \section example_sec Caller Example
*
* For an example "main" that invokes the libGaussMix API routines, see sample_main.cpp.
*
* \section Notes
*
* 1. Per the first reference, the values of the Gaussian density functions may be so small as to underflow to zero, and therefore, it is desirable to perform
* the EM algorithm in the log domain. This implementation takes that approach.
*
* 2. libGaussMix supports csv and svm-style data formats (c.f. www.csie.ntu.edu.tw/~cjlin/libsvm/). Sample files
* (multid_data_1.csv and multid_data_2.svm, respectively) are provided for convenience. The former consists of 20
* three-dimensional data points falling into two clusters centered on (1,1,1) and (100,100,100). The second is similarly
* distributed, with a "+1" cluster centered on (10,10,10) and "-1" cluster centered on (50,50,50).
*
* 3. The Matrix.h/ cpp code wraps LAPACK routines. To work with these routines efficiently, the Matrix class maintains
* internal buffers that may be modified by operations that, from a semantic point of view, are read-only. Hence many
* libGauss routines take non-const Matrix arguments, where const Matrix arguments would be typically be used.
*
* 4. For compilers that support it, the code is configured to use open mp (www.openmp.org) to parallelize various for loops,
* where the loop is over the cluster of the Gaussian Mixture Model.
* .
* \section References
*
*  1. Press, et. al., Numerical Recipes 3rd Edition: The Art of Scientific Computing,
*  Cambridge University Press New York, NY, USA ©2007,
*  ISBN:0521880688 9780521880688
*  (c.f. chapter 16).
*
*  2. Douglas A. Reynolds, et. al., "Speaker Verication Using Adapated Gaussian Mixture Models,"
*  Digital Signal Processing 10, 19-24 (2000).
*
*/


//header brick
#include <math.h>
#include <iostream>
#include <ostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <istream>
#include <list>
#include <numeric>
#include <functional>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <set>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

static int myNode=0, totalNodes=1;

#ifdef UseMPI
#include <mpi.h>
#endif /* UseMPI */

//#undef OPENCL

#ifdef OPENCL
#define __NO_STD_VECTOR
#define __NO_STD_STRING
#include <CL/cl.h>
const char* estep_srcfile = "oclEstep.cl";
static char* estep_src = 0;
static int devType = CL_DEVICE_TYPE_GPU;

static cl_int clerr = CL_SUCCESS;
static cl_platform_id cpPlatform = 0;
static cl_device_id device_id = 0;
static cl_context context = 0;
static cl_command_queue commands = 0;
static cl_program program = 0;
static bool isProgBuilt = false;
#endif /* OPENCL */

#include <lapacke.h>

// for kmeans utils
#include "KMeans.h"
 
// for adaptation utils
#include "Adapt.h"

//API header file
#include "GaussMix.h"

// for error handling in C libraries
#include <errno.h>

#define MAX_CLUSTERS 50

//MAX_ITERATIONS is the while loop limiter for Kmeans
#define MAX_ITERATIONS 100

#define BIG_double (INFINITY)

#define MAX_LINE_SIZE 1000

/*
 * SET THIS TO 1 FOR DEBUGGING STATEMENT SUPPORT (via std out)
 */
#define DEBUG 0

using namespace std;

/********************************************************************************************************
 *                         PRIVATE FUNCTION PROTOTYPES
 ********************************************************************************************************/

// EM helper functions
double estep(int n, int m, int k, const double *X,  Matrix &p_nk_matrix, const std::vector<Matrix *> &sigma_matrix, \
                  const Matrix &mu_matrix, const std::vector<double> &Pk_vec);
bool mstep(int n, int m, int k, const double *X, const Matrix &p_nk_matrix, Matrix *sigma_matrix, \
                  Matrix &mu_matrix, std::vector<double> &Pk_vec);
double * matrixToRaw(const Matrix & X);

/******************************************************************************************
 *                             IMPLEMENTATION OF PRIVATE FUNCTIONS
 *******************************************************************************************/
/*! \brief estep is the function that calculates the L and Pnk for a given data point(n) and gaussian(k).
*
@param n number of data points
@param m dimensionality of data
@param k number of clusters
@param X data
@param p_nk_matrix matrix generated by the caller of EM that holds the pnk's calculated
@param sigma_matrix vector of matrix pointers generated by the caller of EM that holds the sigmas calculated
@param mu_matrix matrix of mean vectors generated by caller
@param Pk_vec vector generated by the caller of EM that holds the Pk's calculated
*/
double estep(int n, int m, int k, const double *X,  Matrix &p_nk_matrix, const std::vector<Matrix *> &sigma_matrix,
                    const Matrix &mu_matrix, const std::vector<double> & Pk_vec)
{
    //initialize likelihood
    double likelihood = 0.0;

#ifndef OPENCL    /* non-OpenCL variant */
    //initialize variables
    std::vector<Matrix*> sigma_inverses;
    std::vector<double> determinants;
    for (int gauss = 0; gauss < k; gauss++)
    {
        Matrix *sigma_inv = sigma_matrix[gauss]->inv();
        sigma_inverses.push_back(sigma_inv);

        double determinant = sigma_matrix[gauss]->det();
        determinants.push_back(determinant);
    }

    //for each data point in n
    for (int data_point = 0; data_point < n; data_point++)
    {
        //initialize the x matrix, which holds the data passed in from double *X
        Matrix x(1,m);
        
        //initialize the P_xn to zero to start
        double P_xn = 0.0;

        //for each dimension
        for (int dim = 0; dim < m; dim++)
        {    //put the data stored in the double* in the x matrix you just created
            x.update( X[m*data_point + dim],0,dim );
        }

        //z_max is the maximum cluster weighted density for the data point under any gaussian
        double z_max = 0.0;
        bool z_max_assigned = false;

#ifdef _OPENMP
        #pragma omp parallel for
#endif /* _OPENMP */
        for (int gaussian = 0; gaussian < k; ++gaussian)
        { //initialize the row representation of the mu matrix
            Matrix mu_matrix_row(1,m);
            
            //for each dimension
            for (int dim = 0; dim < m; dim++)
            { //fill in that matrix
                double temp = mu_matrix.getValue(gaussian,dim);
                mu_matrix_row.update(temp,0,dim);
            }

            //(x - mu)
            Matrix* difference_row = x.subtract(mu_matrix_row);
            if (DEBUG)
                difference_row->print();

            //transpose(x - mu)
            Matrix difference_column(m,1);
            for (int i = 0; i < m; i++)
            {    // fill it in
                difference_column.update( difference_row->getValue(0,i), i, 0 );
            }

            //transpose(x - mu) * inv(sigma)
            Matrix* term1 = sigma_inverses[gaussian]->dot( difference_column );
            if (DEBUG)
            {
                std::cout << "sigma_inv dot difference_column" << std::endl;
                term1->print();
            }
            
            //transpose(x - mu) * inv(sigma) * (x - mu)
            Matrix *term2 = difference_row->dot(*term1);
            if (DEBUG)
            {
                std::cout << "term2" << std::endl;
                term2->print();
            }

            //create a double to represent term2, since it's a scalar
            double term2_d = term2->getValue(0,0);
            if( DEBUG )
                printf("Datapoint: %d  Gauss: %d  Term2: %f\n", data_point, gaussian, term2_d);

            // log norm factor is the normalization constant for the density functions
            double log_norm_factor = -0.5*( m*log(2.0*M_PI) + log(determinants[gaussian]) );

            //log density is the log of the density function for the kth gaussian evaluated on the nth data point
            double log_density = log_norm_factor + (-0.5*term2_d);

            //temp1 is the cluster weight for the current gaussian
            double temp1 = Pk_vec[gaussian];

            //temp2 is the log of the cluster weight for the current gaussian
            double temp2 = log(temp1);

            //current z is the log of the density function times the cluster weight
            double current_z = temp2 + log_density;

            //assign current_z
#ifdef _OPENMP
            # pragma omp critical(z_max)
#endif /* _OPENMP */
            if ((z_max_assigned == false) || current_z > z_max)
            {
                z_max = current_z;
                z_max_assigned = true;
            }

            //calculate p_nk = density * Pk / weight
            p_nk_matrix.update(current_z, data_point,gaussian);
            if( DEBUG )
            {
                std::cout << "p_nk_matrix" << std::endl;
                p_nk_matrix.print();
            }

#ifdef _OPENMP
            # pragma omp critical(sigma_inv_tracking)
#endif /* _OPENMP */
            delete difference_row;
            delete term1;
            delete term2;
        } // end gaussian 

        //calculate the P_xn
        for (int gaussian = 0; gaussian < k; gaussian++)
            P_xn += exp( p_nk_matrix.getValue(data_point, gaussian) - z_max );

        //log of total density for data point
        double tempa = log(P_xn);
        double log_P_xn = tempa + z_max;

        //normalize the probabilities per cluster for data point
        for (int gaussian = 0; gaussian < k; gaussian++)
            p_nk_matrix.update( p_nk_matrix.getValue(data_point,gaussian)-log_P_xn, data_point,gaussian );
        
        //calculate the likelihood of this model
        likelihood += log_P_xn;
        if (DEBUG)
            std::cout << "The likelihood for this iteration is " << likelihood << std::endl;
    } // end data_point

#ifdef UseMPI
    // Now reduce the likelihood over all data points:
    double totalLikelihood;
    if (DEBUG)
        std::cout << "Reducing likelihood: " << likelihood << " on node "<< myNode << std::endl;
    MPI_Allreduce(&likelihood, &totalLikelihood, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    likelihood = totalLikelihood;
#endif /* UseMPI */

    for (int i = 0; i < k; i++)
        delete sigma_inverses[i];

#else    /* do OpenCL stuff */
    if( !isProgBuilt )
    {
        ostringstream sout;
        char* prg_opts = 0;

        sout << " -D ESTEP_N=" << n;
        sout << " -D ESTEP_M=" << m;
        sout << " -D ESTEP_K=" << k;
        prg_opts = (char*)(sout.str().c_str());

        clerr = clBuildProgram(program, 0, 0, prg_opts, 0, 0);
        if( clerr != CL_SUCCESS )
        {
            size_t len;
            char buffer[2048];
            clGetProgramBuildInfo( program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            
            std::cerr << clerr << std::endl << buffer << std::endl;
            throw std::runtime_error("Failed to build program");
        }
        else
        {
            isProgBuilt = true;
            delete[] estep_src;
        }
    }

    commands = clCreateCommandQueue(context, device_id, 0, &clerr);
    if( clerr != CL_SUCCESS )
        throw std::runtime_error("Failed to create command queue");
    

    cl_kernel estep_krnl = clCreateKernel( program, "oclEstep", &clerr );
    if( clerr != CL_SUCCESS )
        throw std::runtime_error("Failed to create estep kernel");

    //@NOTE We must use single precision floats here otherwise rely on ocl ext
    // init variables
    cl_mem h_likelihood = 0;

    cl_mem cl_likelihood = 0;
    cl_mem cl_X = 0;
    cl_mem cl_p_nk = 0;
    cl_mem cl_sigma_invs = 0;
    cl_mem cl_dets = 0;
    cl_mem cl_mus = 0;
    cl_mem cl_Pk = 0;
    
    float* pMapLikelihood = 0;
    float* pMapX = 0;
    float* pMapP_nk = 0;
    float* pMapSigmaInvs = 0;
    float* pMapDets = 0;
    float* pMapMus = 0;
    float* pMapPk = 0;
    
    // create host and device buffers
    h_likelihood = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, \
        sizeof(float)*(n), 0, &clerr);

    cl_likelihood = clCreateBuffer(context, CL_MEM_WRITE_ONLY, \
        sizeof(float)*(n), 0, &clerr);
    cl_X = clCreateBuffer(context, CL_MEM_READ_ONLY, \
        sizeof(float)*(n*m), 0, &clerr);
    cl_p_nk = clCreateBuffer(context, CL_MEM_READ_WRITE, \
        sizeof(float)*(n*k), 0, &clerr);
    cl_sigma_invs = clCreateBuffer(context, CL_MEM_READ_ONLY, \
        sizeof(float)*(m*m*k), 0, &clerr);
    cl_dets = clCreateBuffer(context, CL_MEM_READ_ONLY, \
        sizeof(float)*k, 0, &clerr);
    cl_mus = clCreateBuffer(context, CL_MEM_READ_ONLY, \
        sizeof(float)*(k*m), 0, &clerr);
    cl_Pk = clCreateBuffer(context, CL_MEM_READ_ONLY, \
        sizeof(float)*(n*k), 0, &clerr);
    
    // X buffer- init and copy to dev
    pMapX = (float*)clEnqueueMapBuffer(commands, cl_X, CL_TRUE, CL_MAP_WRITE, \
        0, sizeof(float)*(n*m), 0, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to map data points");
    for( int i=0; i<(n*m); ++i )
        pMapX[i] = (float)X[i];
    clerr = clEnqueueUnmapMemObject(commands, cl_X, pMapX, 0, 0, 0);

    // p_nk buffer- init and copy to dev
    pMapP_nk = (float*)clEnqueueMapBuffer(commands, cl_p_nk, CL_TRUE, CL_MAP_WRITE, \
        0, sizeof(float)*(n*k), 0, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to map p_nk");
    for(int i=0; i<n; ++i)
        for(int j=0; j<k; ++j)
            pMapP_nk[i*k+j] = (float)p_nk_matrix.getValue(i,j);
    clerr = clEnqueueUnmapMemObject(commands, cl_p_nk, pMapP_nk, 0, 0, 0);

    // for each gaussian's weight, mu, det(cov), and inv(cov)- init and copy to dev
    pMapSigmaInvs = (float*)clEnqueueMapBuffer(commands, cl_sigma_invs, CL_TRUE, CL_MAP_WRITE, \
        0, sizeof(float)*(m*m*k), 0, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to map sigma inverse");
    pMapDets = (float*)clEnqueueMapBuffer(commands, cl_dets, CL_TRUE, CL_MAP_WRITE, \
        0, sizeof(float)*k, 0, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to map determinant");
    pMapPk = (float*)clEnqueueMapBuffer(commands, cl_Pk, CL_TRUE, CL_MAP_WRITE, \
        0, sizeof(float)*k, 0, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to map Pk");
    pMapMus = (float*)clEnqueueMapBuffer(commands, cl_mus, CL_TRUE, CL_MAP_WRITE, \
        0, sizeof(float)*(k*1*m), 0, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to map mu");

    for( int gMatIdx=0; gMatIdx<k; ++gMatIdx )
    {    // for each distribution cluster
        pMapPk[gMatIdx] = (float)Pk_vec[gMatIdx];    // copy the distribution weight

        Matrix* pMat = sigma_matrix[ gMatIdx ];    // get cov
        pMapDets[ gMatIdx ] = (float)pMat->det();    // ... save det(cov)
        Matrix* pMatInv = pMat->inv();    // ... save inv(cov)
        //flatten mat inv to 1-D array
        for( int rIdx=0; rIdx<m; ++rIdx )
        {
            pMapMus[gMatIdx*m + rIdx] = (float)mu_matrix.getValue(gMatIdx,rIdx);

            for( int cIdx=0; cIdx<m; ++cIdx )
            {
                int idx = (gMatIdx * m*m) + (cIdx*m + rIdx);
                pMapSigmaInvs[idx] = (float)pMatInv->getValue(rIdx,cIdx);
            }
        }

        // the inv() return a heap object, so free it
        delete pMatInv;
    }

    clerr = clEnqueueUnmapMemObject(commands, cl_sigma_invs, pMapSigmaInvs, 0, 0, 0);
    clerr |= clEnqueueUnmapMemObject(commands, cl_dets, pMapDets, 0, 0, 0);
    clerr |= clEnqueueUnmapMemObject(commands, cl_Pk, pMapPk, 0, 0, 0);
    clerr |= clEnqueueUnmapMemObject(commands, cl_mus, pMapMus, 0, 0, 0);
    if( clerr != CL_SUCCESS )
        throw std::runtime_error("failed to unmap gaussian objects");

    // TODO make use of a summation reduction and local scoped vars
    clerr  = clSetKernelArg(estep_krnl, 0, sizeof(cl_mem), &cl_likelihood);
    clerr |= clSetKernelArg(estep_krnl, 1, sizeof(int), &n);
    clerr |= clSetKernelArg(estep_krnl, 2, sizeof(int), &m);
    clerr |= clSetKernelArg(estep_krnl, 3, sizeof(int), &k);
    clerr |= clSetKernelArg(estep_krnl, 4, sizeof(cl_mem), &cl_X);
    clerr |= clSetKernelArg(estep_krnl, 5, sizeof(cl_mem), &cl_p_nk);
    clerr |= clSetKernelArg(estep_krnl, 6, sizeof(cl_mem), &cl_sigma_invs);
    clerr |= clSetKernelArg(estep_krnl, 7, sizeof(cl_mem), &cl_dets);
    clerr |= clSetKernelArg(estep_krnl, 8, sizeof(cl_mem), &cl_mus);
    clerr |= clSetKernelArg(estep_krnl, 9, sizeof(cl_mem), &cl_Pk);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to set kernel arguments");

    // TODO : find better parallelisms to saturate work threads
    size_t globalSize[1];
    size_t localSize[1];
    globalSize[0] = (size_t)(n*k); // each work group is on a data point, and we want k groups
    localSize[0] = (size_t)k; // each work instance is on a cluster
    clerr = clEnqueueNDRangeKernel(commands, estep_krnl, 1, 0, globalSize, localSize, 0, 0, 0);
    if( clerr != CL_SUCCESS )
        throw std::runtime_error("failed to execute kernel");

    // make likelihood values available to host
    pMapLikelihood = (float*)clEnqueueMapBuffer(commands, cl_likelihood, CL_TRUE, CL_MAP_READ, \
        0, sizeof(float)*(n), 0, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("failed to map likelihood");
    if( DEBUG )
    {
        std::cout << "================ LIKELIHOODS " << std::endl;
        for(int i=0; i<n; ++i)
            std::cout << i << " " << pMapLikelihood[i] << std::endl;
    }
    likelihood = 0.0;
    for( int i=0; i<n; ++i )
        likelihood += pMapLikelihood[i];
    clerr  = clEnqueueUnmapMemObject(commands, cl_likelihood, pMapLikelihood, 0, 0, 0);

    // make p_nk values available to host to send to mstep
    pMapP_nk = (float*)clEnqueueMapBuffer(commands, cl_p_nk, CL_TRUE, CL_MAP_READ, \
        0, sizeof(float)*(n*k), 0, 0, 0, &clerr);
    if( clerr != CL_SUCCESS )
        throw std::runtime_error("failed to map p_nk");
    for(int i=0; i<n; ++i)
        for(int j=0; j<k; ++j)
            p_nk_matrix.update( pMapP_nk[i*k+j], i, j);
    clerr = clEnqueueUnmapMemObject(commands, cl_p_nk, pMapP_nk, 0, 0, 0);
    
    // wait for command queue to finish running all kernels
    clerr = clFinish(commands);
    if( clerr != CL_SUCCESS )
        throw std::runtime_error("failed to finish out command queue");

    // release ocl memory buffers
    clerr = clReleaseMemObject(h_likelihood);
    clerr = clReleaseMemObject(cl_likelihood);
    clerr = clReleaseMemObject(cl_X);
    clerr = clReleaseMemObject(cl_p_nk);
    clerr = clReleaseMemObject(cl_sigma_invs);
    clerr = clReleaseMemObject(cl_dets);
    clerr = clReleaseMemObject(cl_mus);
    clerr = clReleaseMemObject(cl_Pk);

    // cleanup OCL kernel
    clerr = clReleaseKernel( estep_krnl );
#endif /* OPENCL */

    if (DEBUG) 
    {
        std::cout << "Likelihood after E-Step: " << likelihood << std::endl;

        std::cout<<"P_nk for itr is";
        p_nk_matrix.print();
    }

    //return the likelihood of this model
    return likelihood;
}

/*! \brief mstep is the function that approximates the mu, sigma and P(k) paramters for a given Gaussian fit.
*
@param n number of data points
@param m dimensionality of data
@param k number of clusters
@param X data
@param p_nk_matrix matrix generated by the caller of EM that holds the pnk's calculated
@param sigma_matrix vector of matrix pointers generated by the caller of EM that holds the sigmas calculated
@param mu_matrix  matrix of mean vectors generated by caller
@param Pk_vec vector generated by the caller of EM that holds the Pk's calculated
*/

bool mstep(int n, int m, int k, const double *X, Matrix &p_nk_matrix, std::vector<Matrix *> &sigma_matrix,
                Matrix &mu_matrix, std::vector<double> & Pk_vec)
{
    // Update Pk_vec and mu_matrix
    int gaussian = 0;
#ifdef _OPENMP
    # pragma omp parallel for
#endif /* _OPENMP */
    for (gaussian = 0; gaussian < k; gaussian++)
    {
        //initialize the mean and covariance matrices that hold the mstep approximations
        Matrix mu_hat(1,m);

        //initialize the normalization factor - this will be the sum of the densities for each data point for the current gaussian
        double norm_factor = 0;

        //initialize the array of doubles of length m that represents the data with some modification
        double x[m];
        //do the mu calculation point by point
        for (int data_point = 0; data_point < n; data_point++)
        {
            // No need to calculate this multiple times
            double p_nk_local = p_nk_matrix.getValue(data_point,gaussian);
            double exp_p_nk = exp(p_nk_local);
            for (int dim = 0; dim < m; dim++)
            {
                x[dim] = X[m*data_point + dim]* exp_p_nk;
            }

            //sum up all the individual mu calculations
            mu_hat.add(x, m, 0);

            //calculate the normalization factor
            norm_factor += exp_p_nk;
        }
        Pk_vec[gaussian] = norm_factor;
        //fill in the mu hat matrix with your new mu calculations, adjusted by the normalization factor
        // Update mu_hat and mu_matrix.  Scale mu_matrix if and only if not using MPI.
        for (int dim = 0; dim < m; dim++)
        {
            mu_matrix.update(mu_hat.getValue(0,dim),gaussian,dim);
        }
    } // parallel for loop over gaussians

    // Reduce and scale PK and mu

    // Temporary space for reduced Pk_vec - also used in calculation of sigma
    double unscaled_Pk_vec[k];
    double global_scale=0.0;

    {
        // Pk
#ifdef UseMPI
        MPI_Allreduce(&(Pk_vec[0]),unscaled_Pk_vec,k,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        for (int i=0; i<k; i++)
            global_scale += unscaled_Pk_vec[i];
        for (int i=0; i<k; i++)
            Pk_vec[i] = unscaled_Pk_vec[i]/global_scale;
#else
        for (int i=0; i<k; i++)
            global_scale += Pk_vec[i];
        for (int i=0; i<k; i++)
        {
            // Fill in unscaled vector, because we didn't reduce into it.
            unscaled_Pk_vec[i] = Pk_vec[i];
            Pk_vec[i] = Pk_vec[i]/global_scale;
        }
#endif /* UseMPI */
        // Mu
#ifdef UseMPI
        // MPI workspace
        double global_work[k*m];
        double local_work[k*m];

        for (int gaussian=0; gaussian<k; gaussian++)
        {
            // extract, already denormalized
            for (int dim = 0; dim < m; dim++)
            {
                local_work[gaussian*m+dim] = mu_matrix.getValue(gaussian,dim);
            }
        }
        // Reduce
        MPI_Allreduce(local_work,global_work,k*m,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        // Scale and restore reduced Mu
        for (int gaussian=0; gaussian<k; gaussian++)
        {
            // replace and normalize
            for (int dim = 0; dim < m; dim++)
            {
                mu_matrix.update(global_work[gaussian*m+dim] / unscaled_Pk_vec[gaussian],gaussian,dim);
            }
        }
#else
        for (int gaussian=0; gaussian<k; gaussian++)
            for (int dim = 0; dim < m; dim++)
                mu_matrix.update(mu_matrix.getValue(gaussian,dim) / unscaled_Pk_vec[gaussian],gaussian,dim);
#endif /* UseMPI */
    }
    // Using new Pk_vec and mu_matrix, calculate updated sigma
    int successflag = 0;
#ifdef _OPENMP
    # pragma omp parallel for
#endif /* _OPENMP */
    for (gaussian = 0; gaussian < k; gaussian++)
    {
        // Check success flag
#ifdef _OPENMP
        #pragma omp flush(successflag)
#endif /* _OPENMP */
        if (successflag == 0)
        {
            Matrix sigma_hat(m,m);

            //calculate the new covariances, sigma_hat
            for (int data_point = 0; data_point < n; data_point++)
            {
                const double *x = &(X[m*data_point]);

                //magical kronecker tensor product calculation
                double pk = exp(p_nk_matrix.getValue(data_point,gaussian));
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < m; j++)
                    {
                        double temp1 = (x[i]-mu_matrix.getValue(gaussian,i)) *
                            (x[j]-mu_matrix.getValue(gaussian,j)) *
                            pk;
                        sigma_hat.update(sigma_hat.getValue(i,j) + temp1, i, j);
                    }
                }
            }//end data point

            //rest of the sigma calculation, adjusted by the normalization factor
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    sigma_hat.update(sigma_hat.getValue(i,j)/(2*Pk_vec[gaussian]),i,j);
                }
            }
            
            //you can't have a negative determinant - if somehow you do, mstep throws up its hands and EM will terminate
            double determinant;
            {
                determinant = sigma_hat.det();
            }
            if (determinant < 0)
            {
                successflag = 1;
#ifdef _OPENMP
            #pragma omp flush(successflag)
#endif /* _OPENMP */
            }

            //assign sigma_hat to sigma_matrix[gaussian]
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    sigma_matrix[gaussian]->update(sigma_hat.getValue(i,j), i, j);
                }
            }

        } // end if successflag == true
#ifdef UseMPI
#ifndef _OPENMP
        // MPI Collectives don't work well inside OpenMP parallel regions.
        // Move this outside the loop.
        {
            int global_successflag;
            
            MPI_Allreduce(&successflag,&global_successflag,1,MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            if (DEBUG)
                std::cout << "global_successflag: "<<global_successflag<<std::endl;
            successflag = global_successflag;
        }        
#endif /* _OPEN_MP */
#endif /* UseMPI */

    } //end gaussian
#ifdef UseMPI
#ifdef _OPENMP
    // Moved outside the parallel loop if using OpenMP and MPI both
    {
        int global_successflag;

        MPI_Allreduce(&successflag,&global_successflag,1,MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (DEBUG)
            std::cout << "global_successflag: "<<global_successflag<<std::endl;
        successflag = global_successflag;
    }        
#endif /* _OPENMP */
#endif /* UseMPI */

    // Reduce  and scale sigma
#ifdef UseMPI
    {
        // MPI workspace
        double global_work[k*m*m];
        double local_work[k*m*m];

        // Sigma
        for (int gaussian=0; gaussian<k; gaussian++)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    local_work[gaussian*m*m+i*m+j] = sigma_matrix[gaussian]->getValue(i,j);
                }
            }

        }
        // Reduce
        MPI_Allreduce(local_work,global_work,k*m*m,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        // Restore and normalize reduced sigma
        for (int gaussian=0; gaussian<k; gaussian++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    sigma_matrix[gaussian]->update(global_work[gaussian*m*m+i*m+j] / unscaled_Pk_vec[gaussian], i, j);
    }
#else
    // Restore and normalize reduced sigma
    for (int gaussian=0; gaussian<k; gaussian++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                sigma_matrix[gaussian]->update( sigma_matrix[gaussian]->getValue(i,j)/ unscaled_Pk_vec[gaussian], i, j );
#endif /* UseMPI */

    if (DEBUG)
    {
#ifdef UseMPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif /* UseMPI */
        std::cout << "Finished M-Step - printing"<<std::endl;

        // Print all the return values: mu, sigma,
        //for (node=0; node < totalNodes; node++)
        for (int node=0; node<totalNodes; node++)
            for (int gaussian=0; gaussian<k; gaussian++)
                for (int dim=0; dim<m ; dim++)
                {
                    double tmp=mu_matrix.getValue(gaussian,dim);
                    if (myNode == node) std::cout << "mu_matrix:  Node: "<<node<<", Gaussian: "<<gaussian<<", dim: "<<dim<<", Value: "<<tmp<<std::endl;
#ifdef UseMPI
                        MPI_Barrier(MPI_COMM_WORLD);
#endif /* UseMPI */
                }
        for (int node=0; node<totalNodes; node++)
            for (int gaussian=0; gaussian<k; gaussian++)
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < m; j++)
                        if (myNode == node)
                            std::cout << "sigma: Node: "<<node<<",Gaussian: "<<gaussian<<", i,j: ("<<i<<", "<<j<<") :"<< sigma_matrix[gaussian]->getValue(i,j)<<std::endl;
#ifdef UseMPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif /* UseMPI */
        for (int node=0; node<totalNodes; node++)
            for (int i=0; i<k; i++)
                std::cout << "Node: "<<myNode<<", Pk_vec["<<i<<"]: "<<Pk_vec[i]<<std::endl;

#ifdef UseMPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif /* UseMPI */
        std::cout << "Finished Printing M-Step"<<std::endl;
    }

    return !successflag;
}



/*******************************************************************************************
 *                         IMPLEMENTATIONS OF PUBLIC FUNCTIONS
 ******************************************************************************************/


int gaussmix::gaussmix_adapt(Matrix & X, int n, vector<Matrix*> &sigma_matrix,
        Matrix &mu_matrix, std::vector<double> &Pks, vector<Matrix*> &adapted_sigma_matrix,
        Matrix &adapted_mu_matrix, std::vector<double> &adapted_Pks)
{
    int result =  gaussmix::adapt(X,n,sigma_matrix,mu_matrix,Pks,adapted_sigma_matrix,adapted_mu_matrix,adapted_Pks);

    return result;
}

double* gaussmix::gaussmix_matrixToRaw(const Matrix & X)
{
    unsigned int rows = X.rowCount();
    unsigned int cols = X.colCount();

    double* ptr = new double[rows * cols];

    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ptr[i*cols + j] = X.getValue(i,j);

    // returns pointer to heap allocated array
    return ptr;
}

int* serializeIntVector(const std::vector<int> &vec)
{
    // +1 to size to store vector size
    int *array = new int[vec.size()+1];
  
    array[0] = vec.size();
    for (unsigned int i=1; i<=vec.size(); ++i)
        array[i] = vec[i-1];

    // returns a pointer to the heap allocated array
    return array;
}

std::vector<int> deserializeIntVector(const int *array)
{
    std::vector<int> vec(array[0]);
    for (int i=0; i<array[0]; i++)
        vec[i] = array[i+1];

    // return a copy of the vector
    return vec;
}

int gaussmix::parse_line(char * buffer, Matrix & X, std::vector<int> & labels, int row, int m)
{
    if (DEBUG)
        std::cout << "Parsing line: " << buffer << std::endl;

    int cols = 0;
    double temp;

    // Reset errno from possible previous issues
    errno = 0;
    if (buffer[MAX_LINE_SIZE - 1] != 0)
    {
        if (DEBUG)
            std::cout << "Max line size exceeded at zero relative line " << row << std::endl;
        return GAUSSMIX_FILE_NOT_FOUND;
    }

    if (strstr(buffer,":"))
    { // we have svm format (labelled data)
        char * plabel = strtok(buffer," ");
        if(plabel)
            sscanf(plabel,"%d",&(labels[row]));

        if (errno != 0)
        {
            if (DEBUG)
                std::cout << "Could not convert label at row " << row << ": " << strerror(errno) << std::endl;
            return GAUSSMIX_FILE_NOT_FOUND;
        }

        // libsvm-style input (label 1:data_point_1 2:data_point_2 etc.)
        for (cols = 0; cols < m; cols++)
        {
            strtok(NULL, ":");    // bump past position label
            sscanf(strtok(NULL, " "), "%lf", &temp);
            if (errno != 0)
            {
                if (DEBUG)
                    std::cout << "Could not convert data at index " << row << " and " << cols << ": " << strerror(errno) << "temp is " << temp << std::endl;
                return GAUSSMIX_FILE_NOT_FOUND;
            }
            X.update(temp,row,cols);
        }
    }
    else
    { // csv-style input (data_point_1,data_point_2, etc.)
        char *ptok = strtok(buffer, ",");
        if (ptok)
            sscanf(ptok, "%lf", &temp);

        if (errno != 0)
        {
            if (DEBUG)
                std::cout << "Could not convert data at index " << row << " and " << cols << ": " << strerror(errno) << "temp is " << temp << std::endl;
            return GAUSSMIX_FILE_NOT_FOUND;
        }

        X.update(temp,row,cols);

        for (cols = 1; cols < m; cols++)
        {
            sscanf(strtok(NULL, ","), "%lf", &temp);
            if (errno != 0)
            {
                if (DEBUG)
                    std::cout << "Could not convert data at index " << row << " and " << cols << ": " << strerror(errno) << "temp is " << temp << std::endl;
                return GAUSSMIX_FILE_NOT_FOUND;
            }
            X.update(temp,row,cols);
        }
    }

    return 0;
}

int gaussmix::gaussmix_parse(char *file_name, int n, int m, Matrix & X, int & localSamples, std::vector<int> & labels )
{
#ifdef UseMPI
    int mpiError = 0;
#endif /* UseMPI */

  // How many samples are local in an MPI run?
    if (totalNodes == 1)
        localSamples = n;
    else if (myNode == 0) 
    {
        int perNode = (n + totalNodes-1)/totalNodes;
        localSamples = n - (totalNodes-1)*perNode;
    }
    else
        localSamples = (n + totalNodes-1)/totalNodes;

    if (myNode == 0)
    { // Read in data on node 0
        FILE *f = fopen(file_name, "r");
        if( 0==f )
            return GAUSSMIX_FILE_NOT_FOUND;

        for (int currentNode=totalNodes-1; currentNode >=0; currentNode--) 
        {
            int rowsToRead, row;
    
            if (totalNodes == 1)
                rowsToRead = n;
            else if (currentNode == 0) 
            {
                int perNode = (n + totalNodes-1)/totalNodes;
                rowsToRead = n - (totalNodes-1)*perNode;
            }
            else
                rowsToRead = (n + totalNodes-1)/totalNodes;
    
            if (DEBUG)
                std::cout << "Reading "<<rowsToRead<<" lines for node "<<currentNode<<std::endl;
    
            X = Matrix(rowsToRead,m);
            labels = std::vector<int>(rowsToRead);
    
            for (row = 0; row < rowsToRead; row++) 
            {
                char buffer[MAX_LINE_SIZE];
                memset(buffer, 0, MAX_LINE_SIZE);
    
                if (fgets(buffer,MAX_LINE_SIZE,f) == NULL)
                {
                    std::cout << "ERROR: Ran out of data on row " << row << std::endl;
                    return GAUSSMIX_FILE_NOT_FOUND;
                }
    
                if (buffer[MAX_LINE_SIZE - 1] != 0)
                {
                    std::cout << "ERROR: Max line size exceeded at zero relative line " << row << std::endl;
                    return GAUSSMIX_FILE_NOT_FOUND;
                }
    
                if (parse_line(buffer, X, labels, row, m) != 0)
                {
                    std::cout << "ERROR: Could not parse line " << row << std::endl;
                    return GAUSSMIX_FILE_NOT_FOUND;
                }
    
                if (DEBUG)
                {
                    int parsedRows = X.rowCount();
                    std::cout << "row is "<<row<<", and X.rowCount is "<<parsedRows<<std::endl;
                }
            } // end for loop to read individual lines
#ifdef UseMPI
            // Send parsed data and labels to node currentNode
            if (DEBUG)
                std::cout << "Read " << row << " rows : " << " of " << rowsToRead << " for node " << currentNode << std::endl;
    
            if (currentNode != 0)
            {
                double *tmp;
                int *itmp;
    
                if (DEBUG)
                    std::cout << "Sending " << row*m << " doubles from node " << myNode << " to " << currentNode << std::endl;
    
                tmp = X.Serialize();
                if (MPI_SUCCESS !=  MPI_Send(tmp, row*m+2, MPI_DOUBLE, currentNode, 99, MPI_COMM_WORLD))
                {
                    std::cout << "Error sending data in MPI"<<std::endl;
                    mpiError = 1;
                }
    
                itmp = serializeIntVector(labels);
                if (MPI_SUCCESS !=  MPI_Send(itmp, row+1, MPI_DOUBLE, currentNode, 100, MPI_COMM_WORLD))
                {
                    std::cout << "Error sending labels in MPI"<<std::endl;
                    mpiError = 1;
                }
            }
#endif /* UseMPI */
        }  // end of for loop over nodes

        fclose(f);
    } // if myNode == 0

#ifdef UseMPI
// Other nodes exist
    else  // not node 0
    {
        // Receive data sent by node 0
        MPI_Status status;
        int matrixSize = localSamples*m+2;
        double tmp[matrixSize];
        int vectorSize = localSamples+1;
        int itmp[vectorSize];

        if (DEBUG)
        {
            std::cout << "Reserved tmp space for comms: "<<matrixSize<<" doubles"<<std::endl;
            std::cout << "Reserved more tmp space for comms: "<<vectorSize<<" ints"<<std::endl;
            std::cout << "Receiving up to "<<matrixSize<<" doubles on node "<<myNode<<std::endl;
        }

        if (MPI_SUCCESS != MPI_Recv(tmp, matrixSize, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status))
        {
            std::cout << "Error receiving data in MPI"<<std::endl;
            mpiError = 1;
        }

        if (DEBUG)
            std::cout << "About to deserialize matrix" << std::endl;

        X.deSerialize(tmp);
        if (MPI_SUCCESS != MPI_Recv(itmp, vectorSize, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &status))
        {
            std::cout << "Error receiving labels in MPI"<<std::endl;
            mpiError = 1;
        }
        labels = deserializeIntVector(itmp);

        if (DEBUG)
            std::cout << "Received" << std::endl;
    }

    if (DEBUG)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        for (int node=0; node<totalNodes; node++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (myNode == node)
            {
                int rows = X.rowCount();
                int cols = X.colCount();
                std::cout << "Read X on node "<<myNode<<": "<<rows<<" x " << cols<<std::endl;
                for (int i=0; i<rows; i++)
                    for (int j=0; j<cols; j++)
                        std::cout << X.getValue(i,j) << " ";
                std::cout << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif /* UseMPI */

    return GAUSSMIX_SUCCESS;
}


double gaussmix::gaussmix_pdf(int m, std::vector<double> X, Matrix &sigma_matrix,std::vector<double> &mu_vector)
{
    const double pi_fac = std::pow(2 * M_PI, m * 0.5);

    // set up our normalizing factor
    double det = sigma_matrix.det();
    double norm_fac = ( 1.0 / (pi_fac * std::pow(det, 0.5)));

    // compute the difference of feature and mean vectors
    double meanDiff[m];

    for (int j = 0; j < m; j++)
        meanDiff[j] = X[j] - mu_vector[j];

    // convert to row and column vector
    Matrix meanDiffRowVec(meanDiff, 1, m, Matrix::ROW_MAJOR);
    Matrix meanDiffColVec(meanDiff, m, 1, Matrix::COLUMN_MAJOR);

    // get inverted covariance matrix
    Matrix * inv = sigma_matrix.inv();

    // get exp of inner product
    // inv is mxm, rowvec is 1xm, colvec is mx1
    Matrix * innerAsMatrix = meanDiffRowVec.dot( *(inv->dot(meanDiffColVec)) );
    double exp_inner = -0.5 * innerAsMatrix->getValue(0,0);


    // roll in weighted sum
    double result = log(norm_fac) +  exp_inner;

    delete inv;
    delete innerAsMatrix;

    return result;
}

double gaussmix::gaussmix_pdf_mix(int m, int k, std::vector<double> X, vector<Matrix*> &sigma_matrix,
        Matrix &mu_matrix, std::vector<double> &Pks)
{
    double sum_probs = 0.0;

    for (int i = 0; i < k; i++)
    {
        std::vector<double> mean_vec;
        mu_matrix.getCopyOfRow(i,mean_vec);
        sum_probs += Pks[i]* \
                exp(gaussmix::gaussmix_pdf(m,X,*(sigma_matrix[i]),mean_vec));
    }

    return log(sum_probs);
}

int gaussmix::gaussmix_train(int n, \
                 int m, \
                 int k, \
                 int max_iters, \
                 Matrix & Y, \
                 vector<Matrix*> &sigma_matrix,\
                 Matrix &mu_matrix, \
                 std::vector<double> &Pks, \
                 double * op_likelihood)
{
    clock_t start = clock();
    double * X = gaussmix::gaussmix_matrixToRaw(Y);

    //epsilon is the convergence criteria - the smaller epsilon, the narrower the convergence
    double epsilon = 0.001;

    //initialize iteration counter
    int counter = 0;

    // for return code
    int condition = GAUSSMIX_SUCCESS;

    //initialize the p_nk matrix
    Matrix p_nk_matrix(n,k);

    //initialize likelihoods to zero
    double new_likelihood = 0.;    
    double old_likelihood = 0.;
    
    //take the cluster centroids from kmeans as initial mus 
    double *kmeans_mu = gaussmix::kmeans(m, X, n, k);
    
    //if you don't have anything in kmeans_mu, the rest of this will be really hard
    if ( 0 == kmeans_mu )
    {
        delete[] X;
        if (DEBUG)
            std::cout << "Error: kmeans_mu is empty"<<std::endl;

        return std::numeric_limits<double>::infinity();
    }

    //initialize array of identity covariance matrices, 1 per k
    for(int gaussian = 0; gaussian < k; gaussian++)
        for (int j = 0; j < m; j++)
            sigma_matrix[gaussian]->update( 1.0, j, j );

std::cout<< "initial sigma matrix is Identity matrix of size " << m << std::endl;

    //initialize matrix of mus from kmeans the first time - after this, EM will calculate its own mu's
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++)
            mu_matrix.update( kmeans_mu[i*m + j], i, j );

std::cout << "initial mu matrix";
mu_matrix.print();

    //initialize Pks
    Pks.clear();
    Pks.resize( k, 1.0/k );

std::cout << "initial Pk vector vals set to " << 1.0/k << std::endl;

std::cout << "data matrix" << std::endl;
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            std::cout << " " << X[i*m+j];
        std::cout << std::endl;
    }

    //get a new likelihood from estep to have something to start with
    try
    {
        //printf("test pnk value: %f\n", p_nk_matrix.getValue(0,0));
        //TODO: Need have the ability enforce diagonal sigma ... sum(all elements) > sum(diag())
        new_likelihood = estep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks);
        //printf("new likelihood: %f\n", new_likelihood);
    }
    catch (std::exception e)
    {
        if (DEBUG)
            std::cout << "encountered error " << e.what() << std::endl;

        delete[] X;
        delete[] kmeans_mu;

        // if we can't do first e-step, all bets are off
        return GAUSSMIX_GENERAL_ERROR;
    }
    catch ( ... )
    {
        // if we can't do first e-step, all bets are off
        delete[] X;
        delete[] kmeans_mu;

        return GAUSSMIX_GENERAL_ERROR;
    }

    if (DEBUG)
        std::cout << "new likelihood is " << new_likelihood << std::endl;

    //main loop of EM - this is where the magic happens!
    while ( (fabs(new_likelihood - old_likelihood) > epsilon) && (counter < max_iters))
    {
        if (DEBUG)
        {
            std::cout << "new likelihood is " << new_likelihood << std::endl;
            std::cout << "old likelihood is " << old_likelihood << std::endl;
        }

        //store new_likelihood as old_likelihood
        old_likelihood = new_likelihood;

        //here's the mstep exception - if you have a singular matrix, you can't do anything else
        try
        {
            if ( mstep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks) == false)
            {
                if (DEBUG)
                    std::cout << "Found singular matrix - terminated." << std::endl;
                condition = GAUSSMIX_NONINVERTIBLE_MATRIX_REACHED;
                break;
            }
        }
        catch (LapackError e)
        {
            if (DEBUG)
                std::cout << "Found lapacke error: " << e.what() << std::endl;

            if (counter >= 1)
            {
                // able to do at least 1 EM cycle
                condition = GAUSSMIX_NONINVERTIBLE_MATRIX_REACHED;
                break;
            }
            else
            {
                delete[] X;
                delete[] kmeans_mu;
                return GAUSSMIX_GENERAL_ERROR;
            }
        }
        catch (...)
        {
            delete[] X;
            delete[] kmeans_mu;
            return GAUSSMIX_GENERAL_ERROR;
        }
        
        //run estep again to get a new likelihood
        new_likelihood = estep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks);
        
        //increment the counter
        counter++;
    }    // EM algo's while-loop

    delete[] X;
    delete[] kmeans_mu;

    if (DEBUG)
    {
        std::cout << "last new likelihood is " << new_likelihood << std::endl;
        std::cout << "last old likelihood is " << old_likelihood << std::endl;

        //tell the user how many times EM ran
        std::cout << "Total number of iterations completed by the EM Algorithm is: " << counter << std::endl;
    }
    
    *op_likelihood =  new_likelihood;

    if (condition >= 0)
    { // no convergence or convergence?
        condition = (counter == max_iters ? GAUSSMIX_MAX_ITERS_REACHED : GAUSSMIX_SUCCESS);
    }

    clock_t end = clock();
    std::cout << "Elapsed time: " << (double)(end-start)/CLOCKS_PER_SEC << " seconds" << std::endl;

    return condition;
}

void gaussmix::init(int *argc, char ***argv)
{
#ifdef OPENCL
    clerr = clGetPlatformIDs(1, &cpPlatform, 0);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("Failed to find platform");
    
    clerr = clGetDeviceIDs(cpPlatform, devType, 1, &device_id, 0);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("Failed to create device group");

    context = clCreateContext(0, 1, &device_id, 0, 0, &clerr);
    if(clerr != CL_SUCCESS)
        throw std::runtime_error("Failed to create context");

    FILE* pFileStream = 0;
    pFileStream = fopen( estep_srcfile, "rb" );
    if( 0 == pFileStream )
        throw std::runtime_error("Failed to open opencl src file");

    fseek(pFileStream, 0, SEEK_END);
    size_t szFileSize = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET);

    estep_src = new char[szFileSize +1];
    size_t bytesRead = fread(estep_src, szFileSize, sizeof(char), pFileStream);
    if( bytesRead != 1)
    {
        fclose( pFileStream );
        delete[] estep_src;
        throw std::runtime_error("Failed to read opencl src file");
    }
    fclose( pFileStream );
    estep_src[szFileSize]='\0';

    program = clCreateProgramWithSource(context, 1, (const char**)&estep_src, &szFileSize, &clerr);
    if( clerr != CL_SUCCESS )
        throw std::runtime_error("Failed to create program from opencl src");

    isProgBuilt = false;
#elif UseMPI
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNodes); 
    MPI_Comm_rank(MPI_COMM_WORLD, &myNode);
    if (myNode == 0)
    {    // master node output only
        std::cout << "Using MPI with size " << totalNodes << std::endl;
    }
#endif /* OPENCL , UseMPI */
}

void gaussmix::fini()
{
#ifdef OPENCL
    // TODO release mem objects and kernels as needed
    clReleaseCommandQueue(commands);
    clReleaseProgram(program);
    clReleaseContext(context);
#elif UseMPI
    MPI_Finalize();
#endif /* UseMPI */
}

