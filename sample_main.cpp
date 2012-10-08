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


/*! \file sample_main.cpp
*   \brief sample main file to show exercise of EM routines.
*/
#include <stdio.h>
#include <vector>
#include <iostream>
#include "EM_Algorithm.h"

using namespace std;

int main(int argc, char *argv[])
{

	// initialize variables
	int i, k, m, n;
	// throw an error if the command line arguments don't match ParseCSV's inputs
	if (argc != 5)
	{
		cout << " Usage: em_algorithm <data_file> <num_dimensions> <num_data_points> <num_clusters>" << endl;
		return 1;
	}
	int errno = 0;

	// take in the command line arguments and assign them to the previously initialized variables
	sscanf(argv[2],"%d", &m);
	sscanf(argv[3],"%d", &n);
	sscanf(argv[4],"%d", &k);

	// error checking
	if (errno != 0)
	{
			cout << "Invalid inputs:" << endl;
			cout << " Usage: em_algorithm <data_file> <num_dimensions> <num_data_points> <num_clusters>" << endl;
			return 1;
	}
	// reading in and parsing the data
	// create an array of doubles that is n by m dimensional
	double * data = new double[n*m];

	if (ParseCSV(argv[1], data, n, m) != 1)
	{
			cout << "Invalid input file; must be csv, one sample per row, data points as floats" << endl;
			return 1;
	}

	// create vectors that hold pointers to the EM result covariance matrices
	vector<Matrix *> sigma_vector;
	for (i = 0; i < k; i++)
	{
			Matrix*p = new Matrix(m,m);
			sigma_vector.push_back(p);
	}
	//create mean and Pk matrices for EM to fill
	Matrix mu_local(k,m);
	Matrix Pk_matrix(1,k);

	// run the EM function
	double log_likelihood = 0;
	try
	{
		log_likelihood = EM(n, m, k, data, sigma_vector, mu_local, Pk_matrix);

		//print out results
		cout << "The matrix of mu's approximated by the EM algorithm is " << endl;
		mu_local.print();

		cout << "The matrix of Pk's approximated by the EM algorithm is " << endl;
		Pk_matrix.print();

		cout << "The log likelihood (density) of the data is " << log_likelihood << endl;
	}
	catch (...)
	{
		cout << "error encountered during processing " << endl;
	}

    delete[] data;
    mu_local.clear();
    Pk_matrix.clear();

    return 0;
}
