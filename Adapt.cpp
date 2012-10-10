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


/*! \file Adapt.cpp
*   \brief implementations for GMM adaptation
*/

#include <syslog.h>
#include <vector>
#include <exception>

#include "Adapt.h"


using namespace std;


/********************************************************************************************************
 * 						PRIVATE FUNCTION PROTOTYPES
 ********************************************************************************************************/
int compute_new_means(const Matrix & mu_matrix,const Matrix & weighted_means,const vector<double> & alphas,
						Matrix & adapted_mu_matrix);

int compute_new_weights(const vector<double> & alphas,const vector<double> & norm_constants,
		const Matrix & Pks, Matrix & new_weights);

int compute_norm_constants(const Matrix & posteriors,vector<double> & norm_constants);

int compute_posteriors(const double * X, int n, int m, const Matrix & mu_matrix,
		const vector<Matrix *> & sigma_matrix, const Matrix & Pks, Matrix & posteriors);

int compute_weighted_means(const double * X,const Matrix & posteriors,const vector<double> & norm_constants,
		Matrix &  weighted_means);

/*************************************************************************************************************
 *                            PRIVATE FUNCTIONS
 **************************************************************************************************************/

/*! \brief compute_new_covariances
 *
 * @param[in] mu_matrix matrix of old cluster means
 * @param[in] sigma_matrix vector of (ptrs to) old covariance matrices
 * @param[in] alphas the alpha constants used for weight computations
 * @param[in] adapted_means the new cluster means returned from compute_new_means
 * @param[out] adapted_sigma_matrix (ptrs to) the new covariance matrices
 * @return 1 on success, 0 on error
 */
int compute_new_covariances(const Matrix & mu_matrix, const vector<Matrix * > & sigma_matrix,
						const vector<double> & alphas,const Matrix & adapted_means,
						vector <Matrix *> & adapted_sigma_matrix)
{
	int retcode = 0;

	try
	{
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
 * @param[in] mu_matrix matrix of old cluster means
 * @param[in] the mean vectors weighted by the posteriors
 * @param[in] alphas the alpha constants to use in the weight computation
 * @param[out] adapted_mu_matrix the new cluster means
 * @return 1 on success, 0 on error
 */
int compute_new_means(const Matrix & mu_matrix,const Matrix & weighted_means,const vector<double> & alphas,
						Matrix & adapted_mu_matrix)
{
	int retcode = 0;

	try
	{
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
 * @param[in] alphas the alpha constants to use in the weight computation
 * @param[in] norm_constants the normalization constants (i-th constant is for i-th cluster)
 * @param[in] Pks cluster weights as a 1 X (num_clusters) Matrix
 * @param[out] new_Pks the new cluster weights as a 1 X (num_clusters) Matrix
 * @return 1 on success, 0 on error
 */
int compute_new_weights(const vector<double> & alphas,const vector<double> & norm_constants,
		const Matrix & Pks, Matrix & new_weights)
{
	int retcode = 0;

	try
	{
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
 *	@param[in] posteriors the matrix of posterior densities
 *	@param[out] norm_constants the normalization constants (i-th constant is for i-th cluster)
 *	@return 1 on success 0 on error
 */
int compute_norm_constants(const Matrix & posteriors,vector<double> & norm_constants)
{
	int retcode = 0;

	try
	{
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
 * @param[in] X n by m array of data points
 * @param[in] n the number of data points
 * @param[in] mu_matrix matrix of cluster means returned from EM call (EM_Algorithm.h)
 * @param[in] sigma_matrix vector of pointers to covariances matrices returned from EM call
 * @param[in] Pks cluster weights returned from EM call
 * @param[out] posteriors n by k matrix in which posterior densities will be placed, where k is the number of clusters
 * @return 1 on success, 0 on error
 */
int compute_posteriors(const double * X, int n, Matrix & mu_matrix, vector<Matrix *> & sigma_matrix,
						Matrix & Pks, Matrix & posteriors)
{
	int retcode = 0;

	try
	{
		retcode = 1;
	}
	catch (exception e)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute posteriors resulted in %s: ",e.what());
	}
	catch (...)
	{
		syslog(LOG_WARNING,"gaussmix: attempt to compute posteriors resulted in unknown error");
	}

	return retcode;
}

/*! \brief compute_weighted_means compute the mean vectors weighted by the posteriors
 *
 * @param[in] X n by m array of data points
 * @param[in] posteriors n by k matrix in which posterior densities will be placed, where k is the number of clusters
 * @param[in] norm_constants the normalization constants (i-th constant is for i-th cluster)
 * @param[out] weighted_means the mean vectors weighted by the posteriors
 * @return 1 on success, 0 on error
 *
 */
int compute_weighted_means(const double * X,const Matrix & posteriors,const vector<double> & norm_constants,
		Matrix &  weighted_means)
{
	int retcode = 0;

	try
	{
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


/*! \brief compute_expected_squares compute the squared-mean vectors weighted by the posteriors
 *
 * @param[in] X n by m array of data points
 * @param[in] posteriors n by k matrix in which posterior densities will be placed, where k is the number of clusters
 * @param[in] norm_constants the normalization constants (i-th constant is for i-th cluster)
 * @param[out] expected_squares the mean-square vectors weighted by the posteriors
 * @return 1 on success, 0 on error
 *
 */
int compute_expected_squares(const double * X,const Matrix & posteriors,const vector<double> & norm_constants,
		Matrix &  expected_squares)
{
	int retcode = 0;

	try
	{
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

/******************************************************************
 *                        IMPLEMENTATIONS OF PUBLIC FUNCTIONS
 ******************************************************************/


int gaussmix::adapt(const double *X, int n, vector<Matrix*> &sigma_matrix,
		Matrix &mu_matrix, Matrix &Pks,
		vector<Matrix*> &adapted_sigma_matrix, Matrix &adapted_mu_matrix, Matrix &adapted_Pks)
{
	int num_clusters = mu_matrix.rowCount();		// number of gaussians in mix
	int num_dimensions = mu_matrix.colCount();		// number of data dimensions

	/*
	 * 1. construct an n X k "posterior" matrix of weighted density values p_nk = P(n|k)*P(k)/Q_n
	 * for each data point, where Q_n is the normalization factor given by the sum of P(n|k)*P(k) over all k.
	*/
	Matrix posteriors(n,Pks.colCount());
	if (compute_posteriors(X,n,mu_matrix,sigma_matrix,Pks,posteriors) == 0)
	{
		return 0;
	}

	/*
	 *  2. now get a normalization constants v_i for each cluster by summing the normalized densities for each
	 *  data point, under the cluster.
	 */
	vector<double> norm_constants;
	if (compute_norm_constants(posteriors,norm_constants) == 0)
	{
		return 0;
	}

	/*
	 *  3. now compute the "alpha" constants a_i as v_i/ (v_i + relevance_factor).
	 */
	const int relevance_factor = 16;	// see ref 2 in doxygen main index page
	std::vector<double> alphas;
	for (int i = 0; i < num_clusters; i++)
	{
		alphas.push_back(norm_constants[i] / ( norm_constants[i] + relevance_factor));

	}

	/*
	 *  4. now for each cluster, compute the cluster mean e_i as the weighted sum of the data point vectors,
	 *  where the (n,k) posterior matrix entry is the weight for data point n and cluster k, and the expectation
	 *  has normalizing constant v_i.
	 */
	Matrix weighted_means(num_clusters,num_dimensions);
	if (compute_weighted_means(X,posteriors,norm_constants,weighted_means) == 0)
	{
		return 0;
	}

	/*
	 *  5. now compute the "expected squares" matrix E_i for each cluster k, where by "expected square"
	 *  we mean the expected value of the diagonal matrix whose (i,i)-th entry is given by the square of
	 *  the i-th component of a randomly chosen data point n with itself, weighted by the (n,k) entry of the posterior
	 *  matrix, and the expectation has normalizing constant v_i.
	 *
	 */
	Matrix expected_squares(num_clusters,num_dimensions);
	if (compute_expected_squares(X,posteriors,norm_constants,weighted_means) == 0)
	{
		return 0;
	}

	/*
	 *  6. now compute the new cluster weights W_i = Y [(a_i * v_i/N) + (1-a_i)*w_i
	 *  where N is the number of data points, w_i is the old weight, and Y is 1/(sum of all new weights)
	 */
	if (compute_new_weights(alphas,norm_constants,Pks,adapted_Pks) == 0)
	{
		return 0;
	}

	/*
	 *  7. now compute the new cluster means M_i as a_i * e_i + (1-a_i)*m_i where m_i is the old cluster mean
	 */
	if (compute_new_means(mu_matrix,weighted_means,alphas,adapted_mu_matrix) == 0)
	{
		return 0;
	}


	/*
	 *  8. now compute the new covariances C_i as a_i * E_i + (1 - a_i) * ( c_i + diag(m_i) ) - c_i
	 *  where c_i s the old covariance and diag(m_i) is the diagonal matrix w/entry (j,j) given by the
	 *  square of the j-th component of m_i
	 */
	if (compute_new_covariances(mu_matrix,sigma_matrix,alphas,adapted_mu_matrix,adapted_sigma_matrix) == 0)
	{
		return 0;
	}

	return 1;
}


