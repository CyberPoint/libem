/*********************************************************************************
# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
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

/*! \file EM_Algorithm.cpp
    \brief core libGaussMix++ em algorithm method implementations.
    \author Elizabeth Garbee
    \date Summer 2012
*/

/*! \mainpage libGaussMix++: An Expectation Maximization Algorithm for Training Gaussian Mixture Models
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
* the probability density of the point given the cluster. The likelihood density L of the data is the product of these sums, taken over all data points (which are assumed to be independent and identically distributed). The EM algorithm produces an estimate of the P(k) and the P(n|k),
* by attempting to maximize L. Specifically, it yields the estimated means mu(k), covariance matrices sigma(k), and weights P(k) for each Gaussian.
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
* The libGaussMix++ implementation of the EM algorithm uses the "KMeans" clustering algorithm to provide the initial guesses for the means of the K Gaussians, in order to increase the efficiency and efficacy of
* EM. 
*
* One important detail to note is that often, the values of the Gaussian density functions will be so small as to underflow to zero. Therefore, it is very important to work with the logarithms of
* these densities, rather than the densities themselves. This particular implementation works in log space in an attempt to avoid this issue.
*
* \section usage_sec Usage
*
* libGaussMix++ does not actually build as a library. The main training algorithm is provided in EM_Algorithm.h/ EM_Algorithm.cpp. It uses helper routines provided in
* Matrix.h/ Matrix.cpp which in turn wraps LAPACK linear algebra functions. It uses KMeans.h/ KMeans.cpp for initial model parameter guesses.
* To build these files into an executable with a sample driver function provided in sample_main.cpp, follow these steps:
*
* - 1. Install BLAS and LAPACK on your machine if they're not already there (c.f. http://www.netlib.org/lapack/, which bundles a reference vesion of BLAS).
* - 2. Update the environment variables in the libGaussMix++ makefile to point to your environment's BLAS and LAPACK header and library locations.
* - 3. run make on the libGaussMix++ makefile.
* - 4. run the resulting executable via: "em_algorithm <data_file> <num_dimensions> <num_data_points> <num_clusters>". \n
*      Try using one of the sample *.csv or *.svm data files as the first argument: e.g. "em_algorithm multid_data_1.csv 3 20 2".
*
* The libGaussMix++ has been built and tested on SL6.2 using gcc 4.4.5.
* 
*
* \section example_sec Caller Example
*
* For an example "main" that will run EM, see sample_main.cpp.
*
* \section Data Formats
*
* libGaussMix++ supports csv and svm-style data formats (c.f. www.csie.ntu.edu.tw/~cjlin/libsvm/). Sample files
* (multid_data_1.csv and multid_data_2.svm, respectively) are provided for convenience. The former consists of 20
* three-dimensional data points falling into two clusters centered on (1,1,1) and (100,100,100). The second is similarly
* distributed, with a "+1" cluster centered on (10,10,10) and "-1" cluster centered on (50,50,50).
*
* \section References
*
*  Numerical Recipes 3rd Edition: The Art of Scientific Computing,
*  Cambridge University Press New York, NY, USA ©2007,
*  ISBN:0521880688 9780521880688
*  (c.f. chapter 16).
*
*  Douglas A. Reynolds, et. al., "Speaker Verication Using Adapated Gaussian Mixture Models,"
*  Digital Signal Processing 10, 19-24 (2000).
*
*/



//header brick
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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "KMeans.h"

//EM specific header
#include "EM_Algorithm.h"

//#define statements - change #debug to 1 if you want to see EM's calculations as it goes
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
double estep(int n, int m, int k, double *X,  Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix);
bool mstep(int n, int m, int k, double *X, Matrix &p_nk_matrix, Matrix *sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix);


/******************************************************************************************
 * 					         PRIVATE FUNCTIONS
 *******************************************************************************************/
/*! \brief estep is the function that calculates the L and Pnk for a given data point(n) and gaussian(k).
*
@param n number of data points
@param m dimensionality of data
@param k number of clusters
@param X data
@param p_nk_matrix = matrix generated by the caller of EM that holds the pnk's calculated
@param sigma_matrix = vector of matrix pointers generated by the caller of EM that holds the sigmas calculated
@param Pk_matrix = matrix generated by the caller of EM that holds the Pk's calculated
*/

double estep(int n, int m, int k, double *X,  Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix)
{
	//initialize likelihood
	double likelihood = 0;
	
	//initialize variables
	const int pi = 3.141592653;

	std::vector<Matrix *> workingCovars;

	//for each data point in n
	for (int data_point = 0, count = 0; data_point < n; data_point++, count++)
	{
		if (debug)
		{
			cout << "1:beginning iteration " << data_point << " of " << n << endl << flush;
		}


		//initialize the x matrix, which holds the data passed in from double*X
		Matrix x(1,m);
		
		//initialize the P_xn to zero to start
		double P_xn = 0;

		//for each dimension
		for (int dim = 0; dim < m; dim++)
		{	
			if (debug) cout << "trying to assign data " << X[m*dim + data_point] << " to location " << dim << " by " << data_point << endl;

			//put the data stored in the double* in the x matrix you just created
			x.update(X[m*data_point + dim],0,dim);
		}

		//z max is the maximum cluster weighted density for the data point under any gaussian
		double z_max = 0;
		bool z_max_assigned = false;

		//log_densities is the k dimensional array that stores the density in log space
		double log_densities[k];

		//for each cluster
		workingCovars.clear();

		int gaussian = 0;
		#ifdef _OPENMP
		# pragma omp parallel for
		#endif
		for (gaussian = 0; gaussian < k; gaussian++)
		{

			#ifdef _OPENMP
			if (debug)
			{
				cout << "estep: in thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << endl << flush;
			}
			#endif

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

			//sigma^-
			Matrix * sigma_inv;
			//#ifdef _OPENMP
			//# pragma omp critical(lapack)
			//#endif
//#ifdef _OPENMP
//# pragma omp critical(lapack)
//#endif
			sigma_inv = &(sigma_matrix[gaussian]->inv());
			if (debug) cout << "sigma_inv" << endl << flush;
			if (debug) sigma_inv->print();

			//det(sigma)
			double determinant;
			//#ifdef _OPENMP
			//# pragma omp critical(lapack)
			//#endif
//#ifdef _OPENMP
//# pragma omp critical(lapack)
//#endif
			determinant = sigma_matrix[gaussian]->det();


			//make a column representation of the difference in preparation for matrix multiplication
			Matrix difference_column(m,1);
			
			for (int i = 0; i < m; i++)
			{
				//fill it in
				difference_column.update(difference_row.getValue(0,i),i,0);
			}

			//(x - mu) * sigma^-1
			Matrix term1 = sigma_inv->dot(difference_column);
			if (debug) printf("difference_column \n");
			
			//(x - mu) * sigma^-1 * (x - mu)
			Matrix &term2 = difference_row.dot(term1);
			if (debug) printf("term2 \n");
			if (debug) term2.print();

			//create a double to represent term2, since it's a scalar
			double term2_d = 1;
			term2_d = term2.getValue(0,0);
		
			//bringing all the pieces together
			double log_unnorm_density = (-.5 * term2_d);
			double term3 = pow(2*pi, 0.5 * m);
			double term4 = pow(determinant, .5);

			// log norm factor is the normalization constant for the density functions
			double log_norm_factor = log(term3 * term4);

			//log density is the log of the density function for the kth gaussian evaluated on the nth data point
			double log_density = log_unnorm_density - log_norm_factor;

			if (debug) printf("Pk_matrix \n");
			if (debug) Pk_matrix.print();

			//temp1 is the cluster weight for the current gaussian
			double temp1 = Pk_matrix.getValue(0,gaussian);
			if (debug) printf("temp 1 is %lf \n", Pk_matrix.getValue(0,gaussian));

			//temp2 is the log of the cluster weight for the current gaussian
			double temp2 = log(temp1);
			if (debug) printf("temp2 is %lf \n", log(temp1));

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
			workingCovars.push_back(sigma_inv);



		} // end gaussian 

		// free up covars created during above loop
		for (std::vector<Matrix *>::iterator iter = workingCovars.begin(); iter != workingCovars.end();iter++)
		{
			delete *iter;
		}

		for (int gaussian = 0; gaussian < k; gaussian++)
		{
			//calculate the P_xn's
			P_xn += exp(p_nk_matrix.getValue(data_point, gaussian) - z_max);
		}
		if (debug) cout << "P_xn is " << P_xn << endl;

		//log of total density for data point
		double tempa = log(P_xn);
		if (debug) cout << "log Pxn is " << log(P_xn) << endl;
		double log_P_xn = tempa + z_max;
		if (debug) cout << "log Pxn plus z_max is " << log_P_xn << endl;

		for (int gaussian = 0; gaussian < k; gaussian++)
		{
			//normalize the probabilities per cluster for data point
			p_nk_matrix.update(p_nk_matrix.getValue(data_point,gaussian)-log_P_xn,data_point,gaussian);
		}
		
		//calculate the likelihood of this model
		likelihood += log_P_xn;
		if (debug) cout << "The likelihood for this iteration is " << likelihood << endl;
		
	} // end data_point

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
@param Pk_matrix matrix generated by the caller of EM that holds the Pk's calculated
*/

bool mstep(int n, int m, int k, double *X, Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix)
//note: the "hat" denotes an approximation in the literature - I called these matrices <name>_hat to avoid confusion between the estep and mstep calculations
{
	//initialize the Pk_hat matrix
	Matrix Pk_hat(1,k);
	bool successflag = true;

	int gaussian = 0;
	#ifdef _OPENMP
	# pragma omp parallel for
	#endif
	for (gaussian = 0; gaussian < k; gaussian++)
	{	

		#ifdef _OPENMP
		if (debug)
		{
			cout << "mstep: in thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << endl << flush;
		}
		#endif

		//initialize the mean and covariance matrices that hold the mstep approximations
		Matrix sigma_hat(m,m);
		Matrix mu_hat(1,m);

		//initialize the normalization factor - this will be the sum of the densities for each data point for the current gaussian
		double norm_factor = 0;

		//initialize the array of doubles of length m that represents the data
		double x[m];
		
		if (successflag == true)
		{
			//do the mu calculation point by point
			for (int data_point = 0; data_point < n; data_point++)
			{
				for (int dim = 0; dim < m; dim++)
				{
					x[dim] = X[m*data_point + dim]*exp(p_nk_matrix.getValue(data_point,gaussian));
				}

				//sum up all the individual mu calculations
				mu_hat.add(x, m, 0);

				//calculate the normalization factor
				if (debug) cout << "pnk value for norm factor calc is " << p_nk_matrix.getValue(data_point,gaussian) << endl;
				norm_factor += exp(p_nk_matrix.getValue(data_point,gaussian));
				if (debug) cout << "norm factor is " << norm_factor << endl;
			}

			//fill in the mu hat matrix with your new mu calculations, adjusted by the normalization factor
			for (int dim = 0; dim < m; dim++)
			{
				mu_hat.update(mu_hat.getValue(0,dim)/norm_factor,0,dim);
			}
	
			//calculate the new covariances
			for (int data_point = 0; data_point < n; data_point++)
			{
				//fill in x vector for this data_point
				for (int dim = 0; dim < m; dim++)
				{
					x[dim] = X[m*data_point + dim];
				}

				//initialize the x_m matrix
				Matrix x_m(1,m);

				//fill it
				x_m.add(x,m,0);

				//row representation of x - mu for matrix multiplication
				Matrix difference_row = x_m.subtract(mu_hat);

				//column representation of x - mu for matrix multiplication
				Matrix difference_column(m,1);
				if (debug) cout << "difference column:" << endl;
				if (debug) difference_column.print();
				for (int i = 0; i < m; i++)
				{
					//fill it in
					difference_column.update(difference_row.getValue(0,i),i,0);
				}

				//magical kronecker tensor product calculation
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < m; j++)
					{
						double temp1 = difference_row.getValue(0,i) * difference_column.getValue(j,0)*exp(p_nk_matrix.getValue(data_point,gaussian));
						sigma_hat.update(sigma_hat.getValue(i,j) + temp1, i, j);
					}
				}
			}//end data point

			//rest of the sigma calculation, adjusted by the normalization factor
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < m; j++)
				{
					sigma_hat.update(sigma_hat.getValue(i,j)/norm_factor,i,j);
				}
			}
			
			//you can't have a negative determinant - if somehow you do, mstep throws up its hands and EM will terminate
			double determinant;
//			#ifdef _OPENMP
//			# pragma critical(lapack)
//			#endif
			{
				determinant = sigma_matrix[gaussian]->det();
			}
			if (determinant < 0)
			{
				#ifdef __OPENMP
				#pragma omp critical(success_flag)
				#endif
				successflag = false;

			}
			//adjust the Pk_hat calculations by the normalization factor (this particular func is threadsafe)
			Pk_hat.update(norm_factor/n,0,gaussian);
			if (debug) cout << "pk hat matrix" << endl;
			if (debug) Pk_hat.print();

			//assign sigma_hat to sigma_matrix[gaussian]
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < m; j++)
				{
					sigma_matrix[gaussian]->update(sigma_hat.getValue(i,j), i, j);
				}
			}

			//assign mu_hat to mu_matrix
			for (int dim = 0; dim < m; dim++)
			{
				mu_matrix.update(mu_hat.getValue(0,dim),gaussian,dim);

			}

			//assign Pk_hat to Pk_matrix
			Pk_matrix.update(Pk_hat.getValue(0,gaussian),0,gaussian);

		} // end if successflag == true
	} //end gaussian

	//the Pk calculation is a sum - treat it as such
	double sum = 0;
	for (int i = 0; i < k; i++)
	{	
		sum += Pk_matrix.getValue(0,i);
	}

	for (int i = 0; i < k; i++)
	{	
		Pk_matrix.update(Pk_matrix.getValue(0,i)/sum,0,i);
	}

	return successflag;
}



double EM(int n, int m, int k, double *X, vector<Matrix*> &sigma_matrix, Matrix &mu_matrix, Matrix &Pks)

{
	//create an iteration variable
	int iterations;

	//epsilon is the convergence criteria - the smaller epsilon, the narrower the convergence
	double epsilon = .001;

	//initialize a counter
	int counter = 0;

	//initialize the p_nk matrix
	Matrix p_nk_matrix(n,k);

	if (debug) cout << "n is " << n;
	if (debug) cout << "\nm is: " << m << endl;
	if (debug) cout << "created matrix data ... " << endl;
	
	//initialize likelihoods to zero
	double new_likelihood = 0;	
	double old_likelihood = 0;
	
	//take the cluster centroids from kmeans as initial mus 
	if (debug) printf("i will call kmeans \n");
	fflush(stdout);
	double *kmeans_mu = kmeans(m, X, n, k); 
	
	if (debug) printf("i called kmeans \n");
	fflush(stdout);

	//if you don't have anything in kmeans_mu, the rest of this will be really hard
	if (kmeans_mu == 0)
		return std::numeric_limits<double>::infinity();

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
			if (debug) cout << "assigning, k is " << k << ", kmeans_mu[i] is " << kmeans_mu[i] << " at dimensions i (" << i << ") and j (" << j << ")\n";
		}
	}

	//initialize Pks
	for (int gaussian = 0; gaussian < k; gaussian++)
	{
		//all of the Pks have to sum to one
		double term1 = 1.0 / k;
		Pks.update(term1,0,gaussian);
	}

	if (debug) printf("i initialized matrices successfully \n");
	fflush (stdout);

	//get a new likelihood from estep to have something to start with
	new_likelihood = estep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks);
	if (debug) cout << "new likelihood is " << new_likelihood << endl;

	//main loop of EM - this is where the magic happens!
	while (fabs(new_likelihood - old_likelihood) > epsilon)
	{
		if (debug) cout << "new likelihood is " << new_likelihood << endl;
		
		if (debug) cout << "old likelihood is " << old_likelihood << endl;

		//store new_likelihood as old_likelihood
		old_likelihood = new_likelihood;

		//here's the mstep exception - if you have a singular matrix, you can't do anything else
		if ( mstep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks) == false) 
		{
			cout << "Found singular matrix - terminated." << endl;
			break;
		}
		
		//run estep again to get a new likelihood
		new_likelihood = estep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pks);
		
		fflush (stdout);
		
		//brick of sanity checks
		if (debug) cout << " " << counter << " iteration's pnk matrix is " << endl;
		if (debug) p_nk_matrix.print();
		if (debug) cout << " " << counter << " iteration's mu matrix is " << endl;
		if (debug) mu_matrix.print();
		if (debug) cout << " " << counter << " iteration's sigma matrix is " << endl;
		for (int gaussian = 0; gaussian < k; gaussian++)
		{
			for (int j = 0; j < m; j++)
			{
				if (debug) sigma_matrix[gaussian]->print();
			}
		}
		if (debug) cout << " " << counter << " iteration's Pk matrix is " << endl;
		if (debug) Pks.print();

		//increment the counter
		counter++;
	}

	delete[] kmeans_mu;
	if (debug) cout << "last new likelihood is " << new_likelihood << endl;	
	if (debug) cout << "last old likelihood is " << old_likelihood << endl;
	
	//tell the user how many times EM ran
	if (debug) cout << "Total number of iterations completed by the EM Algorithm is \n" << counter << endl;

	return new_likelihood;
}


int ParseCSV(char *file_name, int n, int m, double *data, int * labels )
{
	char buffer[MAX_LINE_SIZE];
	FILE *f = fopen(file_name, "r");
	if (f == NULL)
	{
		cout << "Could not open file" << endl;
		return 0;
	}
	memset(buffer, 0, MAX_LINE_SIZE);
	int row = 0;
	int cols = 0;
	while (fgets(buffer,MAX_LINE_SIZE,f) != NULL)
	{
		if (buffer[MAX_LINE_SIZE - 1] != 0)
		{
			cout << "Max line size exceeded at zero relative line " << row << endl;
			return 0;
		}

		int errno = 0;
		if (char * plabel = strstr(buffer,":"))
		{
			sscanf(plabel,"%d",&(labels[row]));
			if (errno != 0)
			{
				cout << "Could not convert label at row " << row << endl;
				return 0;
			}
			// libsvm-style input (label 1:data_point_1 2:data_point_2 etc.)
			char *ptok = strtok(buffer, " ");	// bump past label
			if (ptok)
			{
				for (cols = 0; cols < m; cols++)
				{
					if (strtok(NULL, ":"))
					{
						sscanf(strtok(NULL, " "), "%lf", &data[row*m + cols]);

						if (errno != 0)
						{
							cout << "Could not convert data at index " << row << " and " << cols << endl;
							return 0;
						}
					}
					else
					{
						cout << "expecting <label>:<data> format at index " << row << " and " << cols << endl;
						return 0;
					}
				}
			}
			else
			{
				cout << "expecting <label><space> at index start of line" << row << endl;
				return 0;
			}
		}
		else
		{
			// csv-style input (data_point_1,data_point_2, etc.)
			char *ptok = strtok(buffer, ",");
			if (ptok) sscanf(ptok, "%lf", &data[row*m]);
			if (errno != 0)
			{
				cout << "Could not convert data at index " << row << " and " << cols << endl;
				return 0;
			}

			for (cols = 1; cols < m; cols++)
			{
				sscanf(strtok(NULL, ","), "%lf", &data[row*m + cols]);

				if (errno != 0)
				{
					cout << "Could not convert data at index " << row << " and " << cols << endl;
					return 0;
				}
			}
		}
		row++;
		memset(buffer, 0, MAX_LINE_SIZE);
	}
	return 1;
}

double pdf(int m, int k, const double *X, const vector<Matrix*> &sigma_matrix,
		const Matrix &mu_matrix, const Matrix &Pks)
{
	return 0.0;
}
