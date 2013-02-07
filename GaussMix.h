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


/*! \file GaussMix.h
*   \brief Function prototypes for libGaussMix++ routines
*   \author Elizabeth Garbee
*   \date Summer 2012
*/
#ifndef GAUSSMIX_H
#define GAUSSMIX_H

#include <limits>
#include <cstring>

#include "Matrix.h"

using namespace std;

namespace gaussmix
{


/************************************************************************************************
** GAUSSMIX FUNCTION "PUBLIC" DECLARATIONS
************************************************************************************************/

/*! \brief gaussmix_adapt: adapt a Gaussian Mixture model to a given sub-population.
*
*
@param[in] X subpopulation data (dimensionality = sigma_matrix.num_cols)
@param[in] n number of data points in sub-pop
@param[in] sigma_matrix vector of covariance matrices from EM call
@param [in] mu_matrix cluster means returned from EM call
@param [in] Pks cluster weights returned by EM call
@param[out] adapted_sigma_matrix vector of covariance matrices (caller allocates)
@param [out] adapted_mu_matrix cluster means (caller allocates)
@param [out] adapted_Pks cluster weights (caller allocates)
@return 1 on success, 0 on error
*/
int gaussmix_adapt(Matrix & X, int n, vector<Matrix*> &sigma_matrix,
		Matrix &mu_matrix, std::vector<double> &Pks, vector<Matrix*> &adapted_sigma_matrix,
		Matrix &adapted_mu_matrix, std::vector<double>& adapted_Pks);

/*! \brief convert the matrix representation of the data to a flat array (caller must delete[]).
 * @param M the matrix (m rows X n cols)
 * @return a ptr to an array A of doubles - first row is A[0] thru A[n-1], second is A[n] thru A[2n -1] etc
 * note: caller is responsible for freeing the memory!
 */
double * gaussmix_matrixToRaw(Matrix & X);

/*! \brief gaussmix_parse: converts csv or svm-format data and converts it to a double array.
*
@param[in] file_name ptr to full file path
@param[in] n the number of data points
@param[in] m dimensionality of the data
@param[out] ref to pre-allocated m X n Matrix (all 0s). data returned here.
@param[out] labels pointer to pre-allocated array of size n, for svm format, or null, for csv format. labels returned here.
@return 1 for success, 0 for an error or failure.
*/
int gaussmix_parse(char *file_name,  int n, int m, Matrix & data, int * labels = NULL);


/*! \brief gaussmix_pdf: compute the log of the  probability of the given data point
*
*
@param[in] m dimensionality of data
@param[in] X data point
@param[in] sigma_matrix covariance matrix for cluster
@param [in] mu_vector  mean for cluster
@return log likelihood
*/
double gaussmix_pdf(int m, std::vector<double> X,Matrix &sigma_matrix,std::vector<double> &mu_vector);


/*! \brief gaussmix_pdf_mix: compute the log of the mixture probability of the given data point
*
*
@param[in] m dimensionality of data
@param[in] k number of clusters
@param[in] X data point
@param[in] sigma_matrix vector of covariance matrices from EM or adpated call
@param [in] mu_matrix cluster means returned from EM or adapted call
@param [in] Pks cluster weights returned by EM or adapted call
@return log likelihood
*/
double gaussmix_pdf_mix(int m, int k, std::vector<double> X, vector<Matrix*> &sigma_matrix,
						Matrix &mu_matrix, std::vector<double> &Pks);




/*! \brief gaussmix_train: train a Gaussian Mixture model on a given data set.
*
*
@param[in] n number of data points
@param[in] m dimensionality of data
@param[in] k number of clusters
@param[out] X data points
@param[out] sigma_matrix vector of matrix pointers generated by the caller of EM that holds the sigmas calculated
@param[out] mu_matrix matrix that holds the mu approximations
@param[out] Pks local copy of the cluster weights generated by the caller of EM that holds the Pk's calculated
@return the log likelihood (density) of the data (or std::numeric_limits::infinity() on error)
*/
double gaussmix_train(int n, int m, int k, Matrix & X, vector<Matrix*> &sigma_matrix,
													Matrix &mu_matrix, std::vector<double>& Pks);




#endif //EM_ALGORITHM_HEADER

}
