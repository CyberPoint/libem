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


/*! \file Adapt.cpp
*   \brief implementations for GMM adaptation
*/

#include <syslog.h>
#include <math.h>
#include <vector>
#include <exception>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef UseMPI
#include "mpi.h"
// Comm handle for nodes with data
static 	MPI_Comm AdaptNodes;
#endif

#include "Adapt.h"
#include "GaussMix.h"  // for gaussmix_pdf()

using namespace std;

#define debug 0

/********************************************************************************************************
 * 						PRIVATE FUNCTION PROTOTYPES
 ********************************************************************************************************/
int compute_expected_squares(Matrix & X,const Matrix & posteriors,const vector<double> & norm_constants,
		std::vector<Matrix *> &  expected_squares);

int compute_new_covariances(const Matrix & mu_matrix, const Matrix & nu_matrix, 
		const vector<Matrix * > & sigma_matrix, const vector<double> & alphas,vector <Matrix *> & expected_squares, vector <Matrix *> & adapted_sigma_matrix);

int compute_new_means(const Matrix & mu_matrix,const Matrix & weighted_means,const vector<double> & alphas,
						Matrix & adapted_mu_matrix);

int compute_new_weights(const vector<double> & alphas,const vector<double> & norm_constants, int num_points,
		const std::vector<double> & Pks, std::vector<double> & new_weights);

int compute_norm_constants(const Matrix & posteriors,vector<double> & norm_constants);

int compute_posteriors(Matrix & X, int n, int m, const Matrix & mu_matrix,
		const vector<Matrix *> & sigma_matrix, const std::vector<double> & Pks, Matrix & posteriors);

int compute_weighted_means(Matrix & X,const Matrix & posteriors,const vector<double> & norm_constants,
		Matrix &  weighted_means);

/*************************************************************************************************************
 *                            PRIVATE FUNCTIONS
 **************************************************************************************************************/

/*! \brief compute_expected_squares compute the squared-mean vectors weighted by the posteriors
 *
 * @param X n by m Matrix of data points
 * @param posteriors n by k matrix in which posterior densities will be placed, where k is the number of clusters
 * @param norm_constants the normalization constants (i-th constant is for i-th cluster)
 * @param[out] expected_squares vector of ptrs to mean-square matrices weighted by the posteriors (caller sets matrices to 0s)
 * @return 1 on success, 0 on error
 *
 *	note: the matrix we returned in the expected value (w.r.t norm constants) of a diagonal matrix
 *	      whose i-th diagonal entry is given by the i-th component of the dot product of a data point with itself
 */
int compute_expected_squares(Matrix & X,const Matrix & posteriors,const vector<double> & norm_constants,
		std::vector<Matrix *> &  expected_squares)
{
	int retcode = 0;

	try
	{
		int num_clusters = norm_constants.size();
		int num_points = posteriors.rowCount();
		int num_dimensions = expected_squares[0]->colCount();

		// for each cluster
		#ifdef _OPENMP
		# pragma omp parallel for
		#endif
		for (int k = 0; k < num_clusters; k++)
		{
			Matrix * pm = expected_squares[k];

			// for each data point
			for (int n = 0; n < num_points; n++)
			{
				// for each dimension
				for (int m = 0; m < num_dimensions; m++ )
				{
					// add in coordinate to running sum
					double val = X.getValue(n,m);
					pm->update(val * val *posteriors.getValue(n,k) + pm->getValue(m,m),m,m);
				}
			}

			// now normalize
			for (int m  = 0; m < num_dimensions; m++)
			{
				pm->update(pm->getValue(m,m)/norm_constants[k],m,m);
			}
		}
#ifdef UseMPI
		for (int k=0; k<num_clusters; k++)
		{
			// Reduce each matrix
			double dim0,dim1;
			double *mat;
			int serialSize = num_clusters*num_clusters+2;
			double global_mat[serialSize];
			
			mat = expected_squares[k]->Serialize();
			// Save dimensions
			dim0 = mat[0];
			dim1 = mat[1];
			MPI_Allreduce(mat, global_mat, serialSize, MPI_DOUBLE, MPI_SUM, AdaptNodes);
			// Restore dimensions after summation
			global_mat[0] = dim0;
			global_mat[1] = dim1;
			expected_squares[k]->deSerialize(global_mat);
		}
#endif
		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute squared means resulted in %s: ",e.what());
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute squared means resulted in unknown error");
	}

	return retcode;
}

/*! \brief compute_new_covariances
 *
 * @param mu_matrix matrix of old cluster means
 * @param nu_matrix matrix of adapated cluster means
 * @param sigma_matrix vector of (ptrs to) old covariance matrices
 * @param alphas the alpha constants used for weight computations
 * @param expected_squares the expected square means returned from compute_expected_squares
 * @param[out] adapted_sigma_matrix (ptrs to) the new covariance matrices (caller inits to 0)
 * @return 1 on success, 0 on error
 */
int compute_new_covariances(const Matrix & mu_matrix, const Matrix & nu_matrix, 
	const vector<Matrix * > & sigma_matrix, const vector<double> & alphas,
	vector <Matrix *> & expected_squares, vector <Matrix *> & adapted_sigma_matrix)
{
	int retcode = 0;

	try
	{

		/*
		 *  now compute the new covariances C_i as a_i * E_i + (1 - a_i) * ( c_i + diag(m_i) ) - diag(m_i)
		 *  where mi_is the old mean, c_i s the old covariance, diag(m_i) is the diagonal matrix w/entry (j,j)
		 *  given by the square of the j-th component of m_i, and E_i is the "expected squares" matrix taken
		 *  w.r.t the subpop to which we're adapting the model.
		 */
		int num_clusters = adapted_sigma_matrix.size();
		int num_dimensions = mu_matrix.colCount();

		// for each cluster
		#ifdef _OPENMP
		# pragma omp parallel for
		#endif
		for (int k = 0; k < num_clusters; k++)
		{
			for (int i = 0; i < num_dimensions; i++)
			{
				for (int j = 0; j < num_dimensions; j++)
				{
					double new_val = alphas[k] * expected_squares[k]->getValue(i,j);
					double old_val = sigma_matrix[k]->getValue(i,j);
					if (i == j)
					{
						double temp = mu_matrix.getValue(k,i);
						old_val += temp*temp;
					}
					old_val *= (1 - alphas[k]);
					if (i == j)
					{
						double temp = nu_matrix.getValue(k,j);
						old_val -= temp*temp;
					}
					adapted_sigma_matrix[k]->update(new_val + old_val,i,j);
				}

			}
		}
		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to adapt covariances resulted in %s: ",e.what());
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to adapt covariances resulted in unknown error");
	}

	return retcode;
}

/*! \brief compute_new_means compute the new cluster means
 *
 * @param mu_matrix matrix of old cluster means
 * @param weighted_means the mean vectors weighted by the posteriors
 * @param alphas the alpha constants to use in the weight computation
 * @param[out] adapted_mu_matrix the new cluster means
 * @return 1 on success, 0 on error
 */
int compute_new_means(const Matrix & mu_matrix,const Matrix & weighted_means,const vector<double> & alphas,
						Matrix & adapted_mu_matrix)
{
	int retcode = 0;


	try
	{

		/*
		 * now compute the new cluster means M_i as a_i * e_i + (1-a_i)*m_i where m_i is the old cluster mean,
		 * a_i is the alpha constant, and e_i is the expected mean under the posterior weights
		 */
		int num_clusters = mu_matrix.rowCount();
		int num_dimensions = mu_matrix.colCount();

		// for each cluster
		#ifdef _OPENMP
		# pragma omp parallel for
		#endif
		for (int k = 0; k < num_clusters;k++)
		{
			for (int m = 0; m < num_dimensions; m++)
			{
				adapted_mu_matrix.update(alphas[k]*weighted_means.getValue(k,m) +
						(1-alphas[k])*mu_matrix.getValue(k,m),k,m);
			}
		}
		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to adapt means resulted in %s: ",e.what());
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to adapt means resulted in unknown error");
	}

	return retcode;
}

/*! \brief compute_new_weights compute the new cluster weights
 * @param alphas the alpha constants to use in the weight computation
 * @param norm_constants the normalization constants (i-th constant is for i-th cluster)
 * @param num_points number of data points
 * @param Pks cluster weights
 * @param[out] new_weights the new cluster weights
 * @return 1 on success, 0 on error
 */
int compute_new_weights(const vector<double> & alphas,const vector<double> & norm_constants, int num_points,
		const std::vector<double> & Pks, std::vector<double> & new_weights)
{
	int retcode = 0;

	try
	{
		/* now compute the new cluster weights W_i = Y [(a_i * v_i/N) + (1-a_i)*w_i
		 *  where v_i are the normalization constants, N is the number of data points,
		 *  w_i is the old weight, and Y is 1/(sum of all new weights)
		 */
		int num_clusters = new_weights.size();

		double sum_weights = 0.0;

		// for each cluster
		#ifdef _OPENMP
		# pragma omp parallel for
		#endif
		for (int k = 0; k < num_clusters; k++)
		{
			double temp = alphas[k] * norm_constants[k]/num_points + (1 - alphas[k])*Pks[k];
			sum_weights += temp;
			new_weights[k] = temp;
		}

		// now re-normalize
		for (int k = 0; k < num_clusters; k++)
		{
			new_weights[k] = new_weights[k]/sum_weights;
		}

		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to adapt weights resulted in %s: ",e.what());
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to adapt weights resulted in unknown error");
	}

	return retcode;
}

/*! \brief compute_norm_constants compute the normalization constants for the posteriors
 *	@param posteriors the matrix of posterior densities
 *	@param[out] norm_constants empty vec of normalization constants (i-th constant is for i-th cluster)
 *	@return 1 on success 0 on error (norm_constants will be populated)
 */
int compute_norm_constants(const Matrix & posteriors,vector<double> & norm_constants)
{
	int retcode = 0;
	int rows = posteriors.rowCount();
	int cols = posteriors.colCount();

	try
	{
		// sum the posteriors for each cluster to obtain the constant for the cluster
#ifdef UseMPI
	  double temp_sum[cols];
	  double global_sum[cols];
	  
	  for (int k = 0; k < cols; k++)
	    {
	      temp_sum[k] = 0.0;
	      for (int n=0; n < rows; n++)
		temp_sum[k] += posteriors.getValue(n,k);
	    }
	  MPI_Allreduce(temp_sum, global_sum, cols, MPI_DOUBLE, MPI_SUM, AdaptNodes);
	  for (int k=0; k < cols; k++)
	    norm_constants.push_back(global_sum[k]);
	  if (debug) cout << "Reduced norm constants"<<endl;
#else
		for (int k = 0; k < cols; k++)
		{
			double temp_sum = 0.0;

			for (int n = 0; n < rows; n++)
			{
				temp_sum += posteriors.getValue(n,k);
			}
			norm_constants.push_back(temp_sum);
		}
#endif
		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute norm constants resulted in %s: ",e.what());
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute norm constants resulted in unknown error");
	}

	return retcode;
}

/*! \brief computer_posteriors compute the poster densities for each data point for each cluster
 *
 * @param X n by m Matrix of data points
 * @param num_points the number of data points
 * @param mu_matrix matrix of cluster means returned from EM call (EM_Algorithm.h)
 * @param sigma_matrix vector of pointers to covariances matrices returned from EM call
 * @param Pks cluster weights returned from EM call
 * @param[out] posteriors n by k matrix in which posterior densities will be placed, where k is the number of clusters
 * @return 1 on success, 0 on error
 */
int compute_posteriors(Matrix & X, int num_points, Matrix & mu_matrix, vector<Matrix *> & sigma_matrix,
						std::vector<double> & Pks, Matrix & posteriors)
{
	int retcode = 0;
	int num_clusters = mu_matrix.rowCount();
	int num_dimensions = mu_matrix.colCount();

	if (debug) cout << "num_clusters: "<<num_clusters<<", num_dimensions: "<<num_dimensions<<", num_points: "<<num_points<<endl;
	try
	{
		// for each cluster
		#ifdef _OPENMP
		# pragma omp parallel for
#endif
		for (int k = 0; k < num_clusters; k++)
		{

			// for each data point
			for (int n = 0; n < num_points; n++)
			{

				// get the log likelihood density for the point
				vector<double> mean_vec;
				mu_matrix.getCopyOfRow(k,mean_vec);

				std::vector<double> temp;
				double lld = gaussmix::gaussmix_pdf(num_dimensions,X.getCopyOfRow(n,temp),
								    *(sigma_matrix[k]),mean_vec);

				// compute the weighted likelihood density (un-log'd)
				double post_prob = exp(lld)*Pks[k];
				posteriors.update(post_prob,n,k);
				if (debug)
				{
					cout << "Printing posteriors in compute_posteriors"<<endl;
					posteriors.print();
				}

			}
		}

		// now for each data point
		for (int n = 0; n < num_points; n++)
		{
			double temp_sum = 0.0;

			// sum the densities for the data point, over the clusters
			for (int k = 0; k < num_clusters; k++)
			{
				temp_sum += posteriors.getValue(n,k);
			}

			// now normalize the data point posteriors by the sum

			// sum the densities for the data point, over the clusters
			for (int k = 0; k < num_clusters; k++)
			{
				posteriors.update(posteriors.getValue(n,k)/temp_sum,n,k);
			}

		}
		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute posteriors resulted in %s: ",e.what());
		if (debug)
		{
			throw(e);
		}
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute posteriors resulted in unknown error");
	}

	return retcode;
}

/*! \brief compute_weighted_means compute the mean vectors weighted by the posteriors
 *
 * @param X n by m matrix of data points (n is number of data points, m is dimensionality)
 * @param posteriors n by k matrix in which posterior densities will be placed, where k is the number of clusters
 * @param norm_constants the normalization constants (i-th constant is for i-th cluster)
 * @param[out] weighted_means the mean vectors weighted by the posteriors (caller inits to 0s matrix of right size)
 * @return 1 on success, 0 on error
 *
 */
int compute_weighted_means(Matrix & X,const Matrix & posteriors,const vector<double> & norm_constants,
		Matrix &  weighted_means)
{
	int retcode = 0;

	try
	{

		int num_clusters = norm_constants.size();
		int num_points = posteriors.rowCount();
		int num_dimensions = weighted_means.colCount();

		// for each cluster
		#ifdef _OPENMP
		# pragma omp parallel for
		#endif
		for (int k = 0; k < num_clusters; k++)
		{
			double temp_vec[num_dimensions];
			for (int m  = 0; m < num_dimensions; m++)
			{
				temp_vec[m] = 0.0; // initialize
			}
			// for each data point
			for (int n = 0; n < num_points; n++)
			{
				// for each dimension
				for (int m = 0; m < num_dimensions; m++ )
				{
					// add in coordinate to running sum
					temp_vec[m] +=  X.getValue(n,m)*posteriors.getValue(n,k);
				}
			}
			// now normalize
			for (int m  = 0; m < num_dimensions; m++)
			{
				temp_vec[m] /= norm_constants[k];
				weighted_means.update(temp_vec[m],k,m);
			}
		}
#if 1
#ifdef UseMPI
		// Reduction and update moved out of OpenMP loop
		  {
		    double global_temp_vec[num_dimensions*num_clusters+2];
		    double *temp_vec = weighted_means.Serialize();
		    double dim0,dim1;
		    dim0 = temp_vec[0];
		    dim1 = temp_vec[1];
		    if (debug) 
		    {
			    cout<<"Matrix Size from dims*clusters: "<<num_dimensions*num_clusters<<", from tem_vec: "<<temp_vec[0]*temp_vec[1]<<endl;
			    cout<<"Matrix dims: "<<dim0<<" by "<<dim1<<endl;
		    }
		    MPI_Allreduce(temp_vec, global_temp_vec, num_dimensions*num_clusters+2, MPI_DOUBLE, MPI_SUM, AdaptNodes);
		    if (debug) cout<<"Constructing matrix of size "<<num_dimensions<<" by "<<num_clusters<<endl;
		    //weighted_means = Matrix(temp_vec);
		    // Dimensions were summed in reduction.  Fix this.
		    global_temp_vec[0] = dim0;
		    global_temp_vec[1] = dim1;
		    weighted_means.deSerialize(global_temp_vec);
		    if (debug) cout<<"Matrix Constructed:"<<endl;
		    if (debug) weighted_means.print();
		  }
#endif
#endif
		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute weighted means resulted in %s: ",e.what());
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute weighted means resulted in unknown error");
	}

	return retcode;
}



/******************************************************************
 *                        IMPLEMENTATIONS OF PUBLIC FUNCTIONS
 ******************************************************************/


int gaussmix::adapt(Matrix & X, int n, vector<Matrix*> &sigma_matrix,
		    Matrix &mu_matrix, std::vector<double> &Pks,
		    vector<Matrix*> &adapted_sigma_matrix,
		    Matrix &adapted_mu_matrix,
		    std::vector<double> &adapted_Pks)
{
	int num_clusters = mu_matrix.rowCount();		// number of gaussians in mix
	int num_dimensions = mu_matrix.colCount();		// number of data dimensions

	int retcode = 1;
	int myNode = 0;
	int nodes = 1;

#ifdef UseMPI
	MPI_Comm_size(MPI_COMM_WORLD, &nodes); 
	MPI_Comm_rank(MPI_COMM_WORLD, &myNode);

	int haveData = (n>0);

	MPI_Comm_split(MPI_COMM_WORLD, haveData, myNode, &AdaptNodes);
#endif
	if (debug) cout << "Adapted data - n is "<<n<<" on node "<<myNode<<endl;

	if (n>0)
	{
		/*
		 * 1. construct an n X k "posterior" matrix of weighted density values p_nk = P(n|k)*P(k)/Q_n
		 * for each data point, where Q_n is the normalization factor given by the sum of P(n|k)*P(k) over all k.
		 */
		Matrix posteriors(n,Pks.size());
		if (debug) cout << "Calculating posteriors on node "<<myNode<<endl;
		retcode = compute_posteriors(X,n,mu_matrix,sigma_matrix,Pks,posteriors);

		if (debug)
		{
			cout << "the posterior on node "<<myNode<<" matrix is: " << endl;
			posteriors.print();
		}

		/*
		 *  2. now get a normalization constants v_i for each cluster by summing the normalized densities for each
		 *  data point, under the cluster.
		 */
		vector<double> norm_constants;
		if (retcode != 0)
		{
			retcode = compute_norm_constants(posteriors,norm_constants);
		}
		if (debug)
		{
			cout << "Computed normalization constants: ";
			for (int i=0; i<num_clusters; i++)
				cout << norm_constants[i] << " ";
			cout << endl;
		}

		/*
		 *  3. now compute the "alpha" constants a_i as v_i/ (v_i + relevance_factor).
		 */
		const int relevance_factor = 16;	// see ref 2 in doxygen main index page
		std::vector<double> alphas;
		if (retcode != 0)
		{
			for (int i = 0; i < num_clusters; i++)
			{
				alphas.push_back(norm_constants[i] / ( norm_constants[i] + relevance_factor));

			}
		}
		if (debug)
		{
			cout << "Computed alpha constants: ";
			for (int i=0; i < num_clusters; i++)
				cout << norm_constants[i] + relevance_factor << " ";
			cout << endl;
		}

		/*
		 *  4. now for each cluster, compute the cluster mean e_i as the weighted sum of the data point vectors,
		 *  where the (n,k) posterior matrix entry is the weight for data point n and cluster k, and the expectation
		 *  has normalizing constant v_i.
		 */
		Matrix weighted_means(num_clusters,num_dimensions);
		if (retcode != 0)
		{
			retcode = compute_weighted_means(X,posteriors,norm_constants,weighted_means);

		}
		if (debug)
		{
			cout << "Computed cluster mean" << endl;
			weighted_means.print();
		}

		/*
		 *  5. now compute the "expected squares" matrix E_i for each cluster k, where by "expected square"
		 *  we mean the expected value of the diagonal matrix whose (i,i)-th entry is given by the square of
		 *  the i-th component of a randomly chosen data point n with itself, weighted by the (n,k) entry of the posterior
		 *  matrix, and the expectation has normalizing constant v_i.
		 *
		 */
		vector<Matrix * > expected_squares;
		if (retcode != 0)
		{
			for (int i = 0; i < num_clusters; i++)
			{
				expected_squares.push_back(new Matrix(num_dimensions,num_dimensions));
			}

			retcode = compute_expected_squares(X,posteriors,norm_constants,expected_squares);

		}
		if (debug)
		{
			cout << "Computed expected squares:";
			for (int i=0; i<num_clusters; i++)
				expected_squares[i]->print();
		}

		/*
		 *  6. now compute the new cluster weights W_i = Y [(a_i * v_i/N) + (1-a_i)*w_i
		 *  where N is the number of data points, w_i is the old weight, and Y is 1/(sum of all new weights)
		 */
		if (retcode != 0)
		{
			retcode = compute_new_weights(alphas,norm_constants,n,Pks,adapted_Pks);
		}
		if (debug)
		{
			cout << "Computed cluster weights" << endl;
		}

		/*
		 *  7. now compute the new cluster means M_i as a_i * e_i + (1-a_i)*m_i where m_i is the old cluster mean
		 */
		if (retcode != 0)
		{
			retcode = compute_new_means(mu_matrix,weighted_means,alphas,adapted_mu_matrix);
		}

		if (debug)
		{
			cout << "Computed new cluster means" << endl;
		}
		/*
		 *  8. now compute the new covariances C_i as a_i * E_i + (1 - a_i) * ( c_i + diag(m_i) ) - c_i
		 *  where c_i s the old covariance and diag(m_i) is the diagonal matrix w/entry (j,j) given by the
		 *  square of the j-th component of m_i and E_i is the "expected squares" matrix
		 */
		if (retcode != 0)
		{
			retcode = compute_new_covariances(mu_matrix,adapted_mu_matrix, sigma_matrix,alphas,expected_squares,adapted_sigma_matrix);
		}

		if (expected_squares.size() > 0)
		{
			for (int i = 0; i < num_clusters; i++)
			{
				delete expected_squares[i];
			}
		}
		if (debug)
		{
			cout << "Computed new covariances" << endl;
		}

	}
#ifdef UseMPI
	// Distribute results to nodes where n == 0
	// Find a node with the results, and distribute from there.
	int masterNode;
	{
		int local_temp=-1;
		if (0<n)
			// We have data locally
			local_temp = myNode;
		MPI_Allreduce(&local_temp, &masterNode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	}
	if (masterNode < 0)
	{
		cout <<"WARNING:  No node had data in this subgroup"<<endl;
	}
	
	// Results: ,
	// vector<Matrix*> &adapted_sigma_matrix
	for (int i=0; i<adapted_sigma_matrix.size(); i++)
	{
		int matrixSize = 2+num_clusters*num_clusters;
		if (myNode == masterNode)
		{
			double *local_temp;
			local_temp = adapted_sigma_matrix[i]->Serialize();
			MPI_Bcast(local_temp, matrixSize, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
		}
		else
		{
			double local_temp[matrixSize];
			MPI_Bcast(local_temp, matrixSize, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
			adapted_sigma_matrix[i]->deSerialize(local_temp);
		}
		
	}
	// Matrix &adapted_mu_matrix
	{
		int matrixSize = 2 + num_clusters*num_dimensions;
		if (myNode == masterNode)
		{
			double *local_temp;
			if (debug)
			{
				cout << "On masterNode "<<masterNode<<", n: "<<n<<", adapted_mu_matrix:"<<endl;
				adapted_mu_matrix.print();
			}
			local_temp = adapted_mu_matrix.Serialize();
			MPI_Bcast(local_temp, matrixSize, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
		}
		else
		{
			double local_temp[matrixSize];
			MPI_Bcast(local_temp, matrixSize, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
			adapted_mu_matrix.deSerialize(local_temp);
		}

	}
	// std::vector<double> &adapted_Pks
	{
		double local_temp[num_clusters];
		if (myNode == masterNode)
		{
			for (int i=0; i<num_clusters; i++)
				local_temp[i] = adapted_Pks[i];
			if (debug) cout << "PK size is "<<adapted_Pks.size()<<endl;
			MPI_Bcast(local_temp, num_clusters, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Bcast(local_temp, num_clusters, MPI_DOUBLE, masterNode, MPI_COMM_WORLD);
			adapted_Pks.resize(num_clusters);
			for (int i=0; i<num_clusters; i++)
				adapted_Pks[i] = local_temp[i];
		}
	}
	if (debug)
	{
		int myGroupSize;
		cout << "MasterNode: "<<masterNode<<endl;
	}
	
	if (debug) cout << "Node "<<myNode<<" finished adapt."<<endl;
	//MPI_Barrier(MPI_COMM_WORLD);
#endif
    return retcode;
}


