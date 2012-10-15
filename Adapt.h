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


/*! \file Adapt.h
*   \brief definitions for GMM adaptation
*/

#ifndef ADAPT_H_
#define ADAPT_H_

#include <vector>

#include "Matrix.h"


namespace gaussmix
{


/*! \brief adapt: adapt a Gaussian Mixture model to a given sub-population.
*
*
@param[in] X subpopulation data (dimensionality = sigma_matrix.num_cols)
@param[in] n number of data points in sub-pop
@param[in] sigma_matrix vector of covariance matrices from EM call
@param [in] mu_matrix cluster means returned from EM call
@param [in] Pks cluster weights returned by EM call
@param[out] adapted_sigma_matrix vector of covariance matrices
@param [out] adapted_mu_matrix cluster means
@param [out] adapted_Pks cluster weight
@return 1 on success, 0 on error
*/
int adapt(const double *X, int n, std::vector<Matrix*> &sigma_matrix,
		Matrix &mu_matrix, Matrix &Pks, std::vector<Matrix*> &adapted_sigma_matrix,
		Matrix &adapted_mu_matrix, Matrix &adapted_Pks);


#endif /* ADAPT_H_ */

}
