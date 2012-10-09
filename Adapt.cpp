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

#include <vector>

#include "EM_Algorithm.h"
#include "Adapt.h"


using namespace std;

/********************************************************************************************************
 * 						PRIVATE FUNCTION PROTOTYPES
 ********************************************************************************************************/
int compute_posteriors(const double * X, int n, int m, const Matrix & mu_matrix,
		const vector<Matrix *> & sigma_matrix, const Matrix & Pks, Matrix & posteriors);

/*************************************************************************************************************
 *                            PRIVATE FUNCTIONS
 **************************************************************************************************************/

/*! \brief computer_posteriors compute the poster densities for each data point for each cluster
 *
 * @param[in] X n by m array of data points
 * @param[in] n the number of data points
 * @param[in] mu_matrix matrix of cluster means returned from EM call (EM_Algorithm.h)
 * @param[in] sigma_matrix vector of pointers to covariances matrices returned from EM call
 * @param[in] Pks cluster weights returned from EM call
 * @param[out] n by k matrix in which posterior densities will be placed, where k is the number of clusters
 * @return 1 on success, 0 on error
 */
int compute_posteriors(const double * X, int n, Matrix & mu_matrix, vector<Matrix *> & sigma_matrix,
						Matrix & Pks, Matrix & posteriors)
{
	return 0;
}


/******************************************************************
 *                        IMPLEMENTATIONS OF PUBLIC FUNCTIONS
 ******************************************************************/


int adapt(int n, const double *X, vector<Matrix*> &sigma_matrix,
		Matrix &mu_matrix, Matrix &Pks,
		vector<Matrix*> &adapted_sigma_matrix, Matrix &adapted_mu_matrix, Matrix &adapted_Pks)
{

	/*
	 * 1. construct an n X k "posterior" matrix of unnormalized weighted density values p_nk = P(n|k)*P(k)
	 * for each data point, with each entry nk normalized by the total density for the n over all k.
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

	/*
	 *  3. now compute the "alpha" constants a_i as v_i/ (v_i + relevance_factor).
	 */

	/*
	 *  4. now for each cluster, compute the cluster mean e_i as the weighted sum of the data point vectors,
	 *  where the (n,k) posterior matrix entry is the weight for data point n and cluster k, and the expectation
	 *  has normalizing constant v_i.
	 */

	/*
	 *  5. now compute the "expected squares" matrix E_i for each cluster k, where by "expected square"
	 *  we mean the expected value of the diagonal matrix whose (i,i)-th entry is given by the square of
	 *  the i-th component of a randomly chosen data point n with itself, weighted by the (n,k) entry of the posterior
	 *  matrix, and the expectation has normalizing constant v_i.
	 *
	 */

	/*
	 *  6. now compute the new cluster weights W_i = Y [(a_i * v_i/N) + (1-a_i)*w_i
	 *  where N is the number of data points, w_i is the old weight, and Y is 1/(sum of all new weights)
	 */

	/*
	 *  7. now compute the new cluster means M_i as a_i * e_i + (1-a_i)*m_i where m_i is the old cluster mean
	 */

	/*
	 *  8. now compute the new covariances C_i as a_i * E_i + (1 - a_i) * ( c_i + diag(m_i) ) - c_i
	 *  where c_i s the old covariance and diag(m_i) is the diagonal matrix w/entry (j,j) given by the
	 *  square of the j-th component of m_i
	 */

	/*
	 *  9. return the W_i, M_i, and C_i and return code 1.
	 */

	return 0;
}

