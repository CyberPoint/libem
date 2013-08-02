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
#include "time.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <set>
#include <exception>
#include <syslog.h>
//#include <cuda.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int myNode=0, totalNodes=1;

#ifdef UseMPI
#include <mpi.h>
#endif

#include <lapacke.h>

// for kmeans utils
#include "KMeans.h"
 
// for adaptation utils
#include "Adapt.h"

//API header file
#include "GaussMix.h"

// for error handling in C libraries
#include <errno.h>

// constants and such
#define sqr(x) ((x)*(x))

#define MAX_CLUSTERS 50

//MAX_ITERATIONS is the while loop limiter for Kmeans
#define MAX_ITERATIONS 100

#define BIG_double (INFINITY)

#define MAX_LINE_SIZE 1000

/*
 * SET THIS TO 1 FOR DEBUGGING STATEMENT SUPPORT (via std out)
 */
#define debug 0

using namespace std;

/********************************************************************************************************
 * 						PRIVATE FUNCTION PROTOTYPES
 ********************************************************************************************************/

// EM helper functions
double estep(int n, int m, int k, double *X,  Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix,
		          Matrix &mu_matrix, std::vector<double> &Pk_vec);
bool mstep(int n, int m, int k, double *X, Matrix &p_nk_matrix, Matrix *sigma_matrix,
		          Matrix &mu_matrix, std::vector<double> &Pk_vec);
double * matrixToRaw(Matrix & X);

/******************************************************************************************
 * 					        IMPLEMENTATION OF PRIVATE FUNCTIONS
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

double estep(int n, int m, int k, double *X,  Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix,
		            Matrix &mu_matrix, std::vector<double> & Pk_vec)
{
	//initialize likelihood
	double likelihood = 0;
	
	//initialize variables
	const int pi = 3.141592653;

	vector<Matrix*> sigma_inverses;
	vector <double> determinants;
	for (int gauss = 0; gauss < k; gauss++) {
		Matrix * sigma_inv;
		sigma_inv = &sigma_matrix[gauss]->inv();
		sigma_inverses.push_back(sigma_inv);

		double determinant;
		determinant = sigma_matrix[gauss]->det();
		determinants.push_back(determinant);
	}

	//for each data point in n
	for (int data_point = 0; data_point < n; data_point++)
	{

		//initialize the x matrix, which holds the data passed in from double*X
		Matrix x(1,m);
		
		//initialize the P_xn to zero to start
		double P_xn = 0;

		//for each dimension

		for (int dim = 0; dim < m; dim++)
		{	
			//put the data stored in the double* in the x matrix you just created
			x.update(X[m*data_point + dim],0,dim);
		}

		//z max is the maximum cluster weighted density for the data point under any gaussian
		double z_max = 0;
		bool z_max_assigned = false;

		int gaussian = 0;
		#ifdef _OPENMP
		# pragma omp parallel for
		#endif
		for (gaussian = 0; gaussian < k; gaussian++)
		{

			//initialize the row representation of the mu matrix
			Matrix mu_matrix_row(1,m);
			
			//for each dimension
			for (int dim = 0; dim < m; dim++)
			{
				//fill in that matrix
				double temp = mu_matrix.getValue(gaussian,dim);
				mu_matrix_row.update(temp,0,dim);
			}
	

			//x - mu
			Matrix& difference_row = x.subtract(mu_matrix_row);
			if (debug) difference_row.print();

			//sigma^ inverse
			//Matrix * sigma_inv;
			//sigma_inv = &(sigma_matrix[gaussian]->inv());
			if (debug) cout << "sigma_inv" << endl << flush;
			//if (debug) sigma_inv->print();

			//det(sigma)
			//double determinant;
			//determinant = sigma_matrix[gaussian]->det();


			//make a column representation of the difference in preparation for matrix multiplication
			Matrix difference_column(m,1);
			
			for (int i = 0; i < m; i++)
			{
				//fill it in
				difference_column.update(difference_row.getValue(0,i),i,0);
			}

			//(x - mu) * sigma^-1
			Matrix &term1 = sigma_inverses[gaussian]->dot(difference_column);
			if (debug) printf("sigma_inv dot difference_column \n");
			if (debug) term1.print();
			
			//(x - mu) * sigma^-1 * (x - mu)
			Matrix &term2 = difference_row.dot(term1);
			if (debug) printf("term2 \n");
			if (debug) term2.print();

			//create a double to represent term2, since it's a scalar
			double term2_d = 1;
			term2_d = term2.getValue(0,0);
			printf("Datapoint: %d  Gauss: %d  Term2: %f\n", data_point, gaussian, term2_d);
			//bringing all the pieces together
			double log_unnorm_density = (-.5 * term2_d);
			double term3 = pow(2*pi, 0.5 * m);
			double term4 = pow(determinants[gaussian], .5);

			// log norm factor is the normalization constant for the density functions
			double log_norm_factor = log(term3 * term4);

			//log density is the log of the density function for the kth gaussian evaluated on the nth data point
			double log_density = log_unnorm_density - log_norm_factor;

			//temp1 is the cluster weight for the current gaussian
			double temp1 = Pk_vec[gaussian];

			//temp2 is the log of the cluster weight for the current gaussian
			double temp2 = log(temp1);

			//current z is the log of the density function times the cluster weight
			double current_z = temp2 + log_density;

			//assign current_z
			#ifdef _OPENMP
			# pragma omp critical(z_max)
			#endif
			if ((z_max_assigned == false) || current_z > z_max)
			{
				z_max = current_z;
				z_max_assigned = true;
			}
			//calculate p_nk = density * Pk / weight
			p_nk_matrix.update(current_z,data_point,gaussian);
			if (debug)
			{
				cout << "p_nk_matrix" << endl;
				p_nk_matrix.print();
				cout << flush;
			}
			#ifdef _OPENMP
			# pragma omp critical(sigma_inv_tracking)
			#endif

			delete &difference_row;
			delete &term1;
			delete &term2;

		} // end gaussian 

		for (int gaussian = 0; gaussian < k; gaussian++)
		{
			//calculate the P_xn's
			P_xn += exp(p_nk_matrix.getValue(data_point, gaussian) - z_max);
		}

		//log of total density for data point
		double tempa = log(P_xn);
		double log_P_xn = tempa + z_max;

		for (int gaussian = 0; gaussian < k; gaussian++)
		{
			//normalize the probabilities per cluster for data point
			p_nk_matrix.update(p_nk_matrix.getValue(data_point,gaussian)-log_P_xn,data_point,gaussian);
		}
		
		//calculate the likelihood of this model
		likelihood += log_P_xn;
		if (debug) cout << "The likelihood for this iteration is " << likelihood << endl;
		
	} // end data_point

#ifdef UseMPI
	// Now reduce the likelihood over all data points:
	double totalLikelihood;
	if (debug) cout << "Reducing likelihood: " << likelihood << " on node "<< myNode << endl;
	MPI_Allreduce(&likelihood, &totalLikelihood, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	likelihood = totalLikelihood;
#endif
	if (debug) 
		cout << "Likelihood after E-Step: " << likelihood << endl;

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

bool mstep(int n, int m, int k, double *X, Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix,
			    Matrix &mu_matrix, std::vector<double> & Pk_vec)
{
	//initialize the Pk_hat matrix
	//note: "hat" denotes an approximation in the literature
	Matrix Pk_hat(1,k);

	// Update Pk_vec and mu_matrix
	int gaussian = 0;
	#ifdef _OPENMP
	# pragma omp parallel for
	#endif
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
#endif
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
#endif
	}
	// Using new Pk_vec and mu_matrix, calculate updated sigma
	int successflag = 0;
	#ifdef _OPENMP
	# pragma omp parallel for
	#endif
	for (gaussian = 0; gaussian < k; gaussian++)
	{
		// Check success flag
#ifdef __OPENMP
#pragma omp flush(successflag)
#endif
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
#ifdef __OPENMP
#pragma omp flush(successflag)
#endif
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
#ifndef __OPENMP
		// MPI Collectives don't work well inside OpenMP parallel regions.
		// Move this outside the loop.
		{
			int global_successflag;
			
			MPI_Allreduce(&successflag,&global_successflag,1,MPI_INT, MPI_LOR, MPI_COMM_WORLD);
			if (debug) cout << "global_successflag: "<<global_successflag<<endl;
			successflag = global_successflag;
		}		
#endif
#endif

	} //end gaussian
#ifdef UseMPI
#ifdef __OPENMP
	// Moved outside the parallel loop if using OpenMP and MPI both
	{
		int global_successflag;

		MPI_Allreduce(&successflag,&global_successflag,1,MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		if (debug) cout << "global_successflag: "<<global_successflag<<endl;
		successflag = global_successflag;
	}		
#endif
#endif

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
	    {
	      for (int i = 0; i < m; i++)
		{
		  for (int j = 0; j < m; j++)
		    {
		      sigma_matrix[gaussian]->update(global_work[gaussian*m*m+i*m+j] / unscaled_Pk_vec[gaussian], i, j);
		    }
		}

	    }
	}
#else
	// Restore and normalize reduced sigma
	for (int gaussian=0; gaussian<k; gaussian++)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				sigma_matrix[gaussian]->update(
					sigma_matrix[gaussian]->getValue(i,j)/ 
					unscaled_Pk_vec[gaussian], i, j);
			}
		}
	}

#endif
	if (debug)
	  {
#ifdef UseMPI
	    MPI_Barrier(MPI_COMM_WORLD);
#endif
	    cout << "Finished M-Step - printing"<<endl;
	  }

	if (debug)
	  {
	    // Print all the return values: mu, sigma,
	    //for (node=0; node < totalNodes; node++)
	    for (int node=0; node<totalNodes; node++)
	      for (int gaussian=0; gaussian<k; gaussian++)
		for (int dim=0; dim<m ; dim++)
		  {
		    double tmp=mu_matrix.getValue(gaussian,dim);
		    if (myNode == node) cout << "mu_matrix:  Node: "<<node<<", Gaussian: "<<gaussian<<", dim: "<<dim<<", Value: "<<tmp<<endl;
#ifdef UseMPI
		    MPI_Barrier(MPI_COMM_WORLD);
#endif
		  }
	    for (int node=0; node<totalNodes; node++)
	      for (int gaussian=0; gaussian<k; gaussian++)
		  for (int i = 0; i < m; i++)
		    for (int j = 0; j < m; j++)
		      if (myNode == node)
			cout << "sigma: Node: "<<node<<",Gaussian: "<<gaussian<<", i,j: ("<<i<<", "<<j<<") :"<< sigma_matrix[gaussian]->getValue(i,j)<<endl;

#ifdef UseMPI
	    MPI_Barrier(MPI_COMM_WORLD);
#endif
	    for (int node=0; node<totalNodes; node++)
	      for (int i=0; i<k; i++)
		cout << "Node: "<<myNode<<", Pk_vec["<<i<<"]: "<<Pk_vec[i]<<endl;

	  }
	if (debug)
	  {
#ifdef UseMPI
	    MPI_Barrier(MPI_COMM_WORLD);
#endif
	    cout << "Finished Printing M-Step"<<endl;
	  }

	return !successflag;
}



/*******************************************************************************************
 * 						IMPLEMENTATIONS OF PUBLIC FUNCTIONS
 ******************************************************************************************/


int gaussmix::gaussmix_adapt(Matrix & X, int n, vector<Matrix*> &sigma_matrix,
		Matrix &mu_matrix, std::vector<double> &Pks, vector<Matrix*> &adapted_sigma_matrix,
		Matrix &adapted_mu_matrix, std::vector<double> &adapted_Pks)
{

	int result =  gaussmix::adapt(X,n,sigma_matrix,mu_matrix,Pks,adapted_sigma_matrix,adapted_mu_matrix,adapted_Pks);
	return result;
}

double * gaussmix::gaussmix_matrixToRaw(Matrix & X)
{
	int rows = X.rowCount();
	int cols = X.colCount();

	double * ptr = new double[rows * cols];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			ptr[i*cols + j] = X.getValue(i,j);
		}
	}

	return ptr;
}

int * serializeIntVector(std::vector<int> vec)
{
  int *array = new int[1+vec.size()];
  
  array[0] = vec.size();
  for (unsigned int i=1; i<= vec.size(); i++)
    array[i] = vec[i-1];
  return array;
}

std::vector<int> deserializeIntVector(int *array)
{
  std::vector<int> vec (array[0]);
  for (int i=0; i< array[0]; i++)
    vec[i] = array[i+1];
  return vec;
}

int gaussmix::parse_line(char * buffer, Matrix & X, std::vector<int> & labels, int row, int m) {
	
  if (debug) cout << "Parsing line: " << buffer << endl;
  int cols = 0;
  double temp;

  // Reset errno from possible previous issues
  errno = 0;
  if (buffer[MAX_LINE_SIZE - 1] != 0)
    {
      if (debug) cout << "Max line size exceeded at zero relative line " << row << endl;
      return GAUSSMIX_FILE_NOT_FOUND;
    }
  if (strstr(buffer,":"))
    {
      // we have svm format (labelled data)
      char * plabel = strtok(buffer," ");
      sscanf(plabel,"%d",&(labels[row]));
      if (errno != 0)
	{
	  if (debug) cout << "Could not convert label at row " << row << ": " << strerror(errno) << endl;
	  return GAUSSMIX_FILE_NOT_FOUND;
	}
      // libsvm-style input (label 1:data_point_1 2:data_point_2 etc.)
      for (cols = 0; cols < m; cols++)
	{
	  strtok(NULL, ":");	// bump past position label
	  sscanf(strtok(NULL, " "), "%lf", &temp);
	  if (errno != 0)
	    {
	      if (debug) cout << "Could not convert data at index " << row << " and " << cols << ": " << strerror(errno) << "temp is " << temp << endl;
	      return GAUSSMIX_FILE_NOT_FOUND;
	    }
	  X.update(temp,row,cols);
	}
    }
  else
    {
      // csv-style input (data_point_1,data_point_2, etc.)
      char *ptok = strtok(buffer, ",");
      if (ptok) sscanf(ptok, "%lf", &temp);
      if (errno != 0)
	{
	  if (debug) cout << "Could not convert data at index " << row << " and " << cols << ": " << strerror(errno) << "temp is " << temp << endl;
				
	  return GAUSSMIX_FILE_NOT_FOUND;
	}
      X.update(temp,row,cols);

      for (cols = 1; cols < m; cols++)
	{
	  sscanf(strtok(NULL, ","), "%lf", &temp);

	  if (errno != 0)
	    {
	      if (debug) cout << "Could not convert data at index " << row << " and " << cols << ": " << strerror(errno) << "temp is " << temp << endl;
	      return GAUSSMIX_FILE_NOT_FOUND;
	    }
	  X.update(temp,row,cols);
	}
    }
  return 0;
}

int gaussmix::gaussmix_parse(char *file_name, int n, int m, Matrix & X, int & localSamples, std::vector<int> & labels )
{
  int mpiError = 0;

  // How many samples are local in an MPI run?
  if (totalNodes == 1) localSamples = n;
  else if (myNode == 0) 
    {
      int perNode = (n + totalNodes-1)/totalNodes;
      localSamples = n - (totalNodes-1)*perNode;
    }
  else localSamples = (n + totalNodes-1)/totalNodes;

  if (myNode == 0)
    {
      // Read in data on node 0

	FILE *f = fopen(file_name, "r");
	if (f == NULL)
	{
		return GAUSSMIX_FILE_NOT_FOUND;
	}
	for (int currentNode=totalNodes-1; currentNode >=0; currentNode--) 
	  {
	    int rowsToRead, row;

	    if (totalNodes == 1) rowsToRead = n;
	    else if (currentNode == 0) 
	      {
		int perNode = (n + totalNodes-1)/totalNodes;
		rowsToRead = n - (totalNodes-1)*perNode;
	      }
	    else rowsToRead = (n + totalNodes-1)/totalNodes;
	    if (debug) cout << "Reading "<<rowsToRead<<" lines for node "<<currentNode<<endl;
	    X = Matrix(rowsToRead,m);
	    labels = std::vector<int> (rowsToRead);
	    for (row = 0; row < rowsToRead; row++) 
	      {
		char buffer[MAX_LINE_SIZE];
		memset(buffer, 0, MAX_LINE_SIZE);
		if (fgets(buffer,MAX_LINE_SIZE,f) == NULL)
		  {
		    cout << "ERROR: Ran out of data on row " << row << endl;
		    return GAUSSMIX_FILE_NOT_FOUND;
		  }
		if (buffer[MAX_LINE_SIZE - 1] != 0)
		  {
		    cout << "ERROR: Max line size exceeded at zero relative line " << row << endl;
		    return GAUSSMIX_FILE_NOT_FOUND;
		  }
	      if (parse_line(buffer, X, labels, row, m) != 0)
		{
		  cout << "ERROR: Could not parse line " << row << endl;
		  return GAUSSMIX_FILE_NOT_FOUND;
		}
	      if (debug)
		{
		  int parsedRows = X.rowCount();
		  cout << "row is "<<row<<", and X.rowCount is "<<parsedRows<<endl;
		}
	    } // end for loop to read individual lines
#ifdef UseMPI
	  // Send parsed data and labels to node currentNode
	  if (debug) 
	    cout << "Read " << row << " rows : " << " of " << rowsToRead << " for node " << currentNode << endl;
	  if (currentNode != 0)
	    {
	      double *tmp;
	      int *itmp;
	      if (debug) cout << "Sending " << row*m << " doubles from node " << myNode << " to " << currentNode << endl;
	      tmp = X.Serialize();
	      if (MPI_SUCCESS !=  MPI_Send(tmp, row*m+2, MPI_DOUBLE, currentNode, 99,
	       				   MPI_COMM_WORLD))
	       	{
	       	  cout << "Error sending data in MPI"<<endl;
	       	  mpiError = 1;
	       	}
	      itmp = serializeIntVector(labels);
	      if (MPI_SUCCESS !=  MPI_Send(itmp, row+1, MPI_DOUBLE, currentNode, 100,
	       				   MPI_COMM_WORLD))
	       	{
	       	  cout << "Error sending labels in MPI"<<endl;
	       	  mpiError = 1;
	       	}
	    }
#endif
	}  // end of for loop over nodes
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

      if (debug) cout << "Reserved tmp space for comms: "<<matrixSize<<" doubles"<<endl;
      if (debug) cout << "Reserved more tmp space for comms: "<<vectorSize<<" ints"<<endl;

      if (debug) cout << "Receiving up to "<<matrixSize<<" doubles on node "<<myNode<<endl;
      if (MPI_SUCCESS != MPI_Recv(tmp, matrixSize, MPI_DOUBLE, 0, 99,
      				  MPI_COMM_WORLD, &status))
      	{
      	  cout << "Error receiving data in MPI"<<endl;
      	  mpiError = 1;
      	}
      if (debug) cout << "About to deserialize matrix" << endl;
      X.deSerialize(tmp);
      if (MPI_SUCCESS != MPI_Recv(itmp, vectorSize, MPI_DOUBLE, 0, 100,
      				  MPI_COMM_WORLD, &status))
      	{
      	  cout << "Error receiving labels in MPI"<<endl;
      	  mpiError = 1;
      	}
      labels = deserializeIntVector(itmp);
      if (debug) cout << "Received" << endl;
    }

  //MPI_Barrier(MPI_COMM_WORLD);

  if (debug)
    {MPI_Barrier(MPI_COMM_WORLD);
      for (int node=0; node<totalNodes; node++)
	{
	  MPI_Barrier(MPI_COMM_WORLD);
	  if (myNode == node)
	    {
	      int rows = X.rowCount();
	      int cols = X.colCount();
	      cout << "Read X on node "<<myNode<<": "<<rows<<" x " << cols<<endl;
	      for (int i=0; i<rows; i++)
		{
		  for (int j=0; j<cols; j++)
		    {
		      cout << X.getValue(i,j) << " ";
		    }
		  cout << endl;
		}
	    }
	  MPI_Barrier(MPI_COMM_WORLD);
	}
    }
#endif
	return GAUSSMIX_SUCCESS;
}


double gaussmix::gaussmix_pdf(int m, std::vector<double> X, Matrix &sigma_matrix,std::vector<double> &mu_vector)
{

	const double pi = 2*acos(0.0);
	const double pi_fac = pow(2 * pi, m * 0.5);

	// set up our normalizing factor
	double det = sigma_matrix.det();
	double norm_fac = ( 1.0 / (pi_fac * pow(det, 0.5)));

	// compute the difference of feature and mean vectors
	double meanDiff[m];

	for (int j = 0; j < m; j++)
	{
		meanDiff[j] = X[j] - mu_vector[j];
	}

	// convert to row and column vector
	Matrix meanDiffRowVec(meanDiff, 1, m, Matrix::ROW_MAJOR);
	Matrix meanDiffColVec(meanDiff, m, 1, Matrix::COLUMN_MAJOR);

	// get inverted covariance matrix
	Matrix & inv = sigma_matrix.inv();

	// get exp of inner product
	// inv is mxm, rowvec is 1xm, colvec is mx1
	Matrix & innerAsMatrix = meanDiffRowVec.dot(inv.dot(meanDiffColVec));
	double exp_inner = -0.5 * innerAsMatrix.getValue(0,0);


	// roll in weighted sum
	double result = log(norm_fac) +  exp_inner;
	delete &innerAsMatrix;
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
		sum_probs += Pks[i]*
				exp(gaussmix::gaussmix_pdf(m,X,*(sigma_matrix[i]),mean_vec));
	}
	return log(sum_probs);

}

int gaussmix::gaussmix_train(int n, int m, int k, int max_iters, Matrix & Y, vector<Matrix*> &sigma_matrix,
									Matrix &mu_matrix, std::vector<double> &Pks, double * op_likelihood)
{
	clock_t start = clock();
	double * X = gaussmix::gaussmix_matrixToRaw(Y);

	//epsilon is the convergence criteria - the smaller epsilon, the narrower the convergence
	double epsilon = .001;

	//initialize iteration counter
	int counter = 0;

	// for return code
	int condition = GAUSSMIX_SUCCESS;

	//initialize the p_nk matrix
	Matrix p_nk_matrix(n,k);

	//initialize likelihoods to zero
	double new_likelihood = 0;	
	double old_likelihood = 0;
	
	//take the cluster centroids from kmeans as initial mus 
	double *kmeans_mu = gaussmix::kmeans(m, X, n, k);
	
	//if you don't have anything in kmeans_mu, the rest of this will be really hard
	if (kmeans_mu == 0)
	{
		delete[] X;
		return std::numeric_limits<double>::infinity();
		if (debug) cout << "Error: kmeans_mu is empty"<<endl;
	}
	//initialize array of identity covariance matrices, 1 per k
	for(int gaussian = 0; gaussian < k; gaussian++)
	{
		for (int j = 0; j < m; j++)
		{
			sigma_matrix[gaussian]->update(1.0,j,j);
			
		}
	}

	//initialize matrix of mus from kmeans the first time - after this, EM will calculate its own mu's
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < m; j++)
		{
			mu_matrix.update(kmeans_mu[i*m + j],i,j);
		}
	}

	//initialize Pks
	for (int gaussian = 0; gaussian < k; gaussian++)
	{
		//all of the Pks have to sum to one
		double term1 = 1.0 / k;
		Pks[gaussian]=term1;
	}

	//get a new likelihood from estep to have something to start with
	try
	{
		//printf("test pnk value: %f\n", p_nk_matrix.getValue(0,0));
		new_likelihood = estep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks);
		//printf("new likelihood: %f\n", new_likelihood);
		
	}
	catch (std::exception e)
	{
		if (debug) cout << "encountered error " << e.what() << endl;

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

	if (debug) cout << "new likelihood is " << new_likelihood << endl;

	//main loop of EM - this is where the magic happens!
	while ( (fabs(new_likelihood - old_likelihood) > epsilon) && (counter < max_iters))
	{
		if (debug) cout << "new likelihood is " << new_likelihood << endl;
		
		if (debug) cout << "old likelihood is " << old_likelihood << endl;

		//store new_likelihood as old_likelihood
		old_likelihood = new_likelihood;

		//here's the mstep exception - if you have a singular matrix, you can't do anything else
		try
		{
			if ( mstep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks) == false)
			{
				if (debug) cout << "Found singular matrix - terminated." << endl;
				condition = GAUSSMIX_NONINVERTIBLE_MATRIX_REACHED;
				break;
			}
		}
		catch (LapackError e)
		{
			if (debug) cout << "Found lapacke error: " << e.what() << endl;

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
	}

	delete[] kmeans_mu;
	if (debug) cout << "last new likelihood is " << new_likelihood << endl;	
	if (debug) cout << "last old likelihood is " << old_likelihood << endl;
	
	//tell the user how many times EM ran
	if (debug) cout << "Total number of iterations completed by the EM Algorithm is \n" << counter << endl;

	delete[] X;
	
	*op_likelihood =  new_likelihood;

	if (condition >= 0)
	{
		// no convergence or convergence?
		condition = (counter == max_iters ? GAUSSMIX_MAX_ITERS_REACHED : GAUSSMIX_SUCCESS);
	}
	clock_t end = clock();
	printf("Elapsed time: %.2f seconds\n", (double)(end - start)/CLOCKS_PER_SEC);
	return condition;
}

void gaussmix::init(int *argc, char ***argv)
{
#ifdef UseMPI
  MPI_Init(argc, argv);

  MPI_Comm_size(MPI_COMM_WORLD, &totalNodes); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myNode);

  if (myNode == 0) printf("Using MPI with size %d\n",totalNodes);
#endif
}

void gaussmix::fini()
{
#ifdef UseMPI
  MPI_Finalize();
#endif
}

