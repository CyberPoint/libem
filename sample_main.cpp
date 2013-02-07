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


/*! \file sample_main.cpp
*   \brief sample main file to show exercise of EM routines.
*/
#include <stdio.h>
#include <vector>
#include <iostream>
#include "GaussMix.h"

using namespace std;

int main(int argc, char *argv[])
{
	/**************************************
	double d[]  = {7.0, 3.2, 4.7 , 1.4};

	std::vector<double> mu_vector;

	mu_vector.push_back(5.93846154);
	mu_vector.push_back(2.70769231);
	mu_vector.push_back(4.26923077);
	mu_vector.push_back(1.30769231);

	Matrix sigma_matrix(4,4);


	sigma_matrix.update(0.25621302,0,0);
	sigma_matrix.update(0.08508876,0,1);
	sigma_matrix.update(0.15426036,0,2);
	sigma_matrix.update(0.05431953,0,3);

	sigma_matrix.update(0.08508876 ,1,0);
	sigma_matrix.update(0.11763314,1,1);
	sigma_matrix.update(0.07792899,1,2);
	sigma_matrix.update(0.03763314,1,3);

	sigma_matrix.update(0.15426036,2,0);
	sigma_matrix.update(0.07792899,2,1);
	sigma_matrix.update(0.14982249,2,2);
	sigma_matrix.update(0.05946746,2,3);

	sigma_matrix.update(0.05431953,3,0);
	sigma_matrix.update(0.03763314,3,1);
	sigma_matrix.update(0.05946746,3,2);
	sigma_matrix.update(0.04071006,3,3);

	gaussmix::gaussmix_pdf(4, d, sigma_matrix,mu_vector);
	**********************************************************/

	// initialize variables
	int i, k, m, n;
	// throw an error if the command line arguments don't match ParseCSV's inputs
	if (argc != 5)
	{
		cout << " Usage: gaussmix <data_file> <num_dimensions> <num_data_points> <num_clusters>" << endl;
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
			cout << " Usage: gaussmix <data_file> <num_dimensions> <num_data_points> <num_clusters>" << endl;
			return 1;
	}
	// reading in and parsing the data
	// create an array of doubles that is n by m dimensional
	Matrix data(n,m);
	int labels[n];

	// initialize labels
	for (int i = 0; i < n; i++)
	{
		labels[i] = 0;
	}

	if (gaussmix::gaussmix_parse(argv[1], n, m, data, labels) != gaussmix::GAUSSMIX_SUCCESS)
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
	std::vector<double> Pk_matrix(k,0.0);

	// run the EM function
	double log_likelihood = 0;
	try
	{
		int ret = gaussmix::gaussmix_train(n, m, k, 100, data, sigma_vector, mu_local, Pk_matrix,&log_likelihood);

		if (ret < gaussmix::GAUSSMIX_SUCCESS)
		{
			cout << "failed to train! - trying building w/debug to see what happened" << endl;

			for (int i = 0; i < k; i++)
			{
				delete sigma_vector[i];
			}
			return -1;
		}

		// print results
		cout << "The matrix of Pk's approximated by the EM algorithm is " << endl;

		std::string s;
		for (int i = 0; i < k; i++)
		{
			s += Pk_matrix[i];
			s += " ";
		}
		cout << s << endl;


		cout << "The matrix of mu's approximated by the EM algorithm is " << endl;
		mu_local.print();

		for (int i = 0; i < k; i++)
		{
			cout << "Covariance matrix " << i << " approximated by the EM algorithm is " << endl;
			sigma_vector[i]->print();
		}

		cout << "The log likelihood (density) of the data is " << log_likelihood << endl;

		// now let's restrict to the -1 subpopulation, if we have labels
		if (labels[0] != 0)  // assume 0 indicates absence of labels
		{
			int num_subpop = 0;

			// isolate subpopulation
			Matrix S;
			double temp[n];
			for (int i = 0; i < n; i++)
			{
				if (labels[i] == -1)
				{
					for (int j = 0; j < m; j++)
					{
						temp[j] = data.getValue(i,j);
					}
					S.insertRow(&temp[0],n,i);
					num_subpop++;
				}
			}
			// create vectors that hold pointers to the adapted result covariance matrices
			vector<Matrix *> adapted_sigma_vector;
			for (int i = 0; i < k; i++)
			{
					Matrix*p = new Matrix(m,m);
					adapted_sigma_vector.push_back(p);
			}
			//create mean and Pk matrices for adapt rtn to fill
			Matrix adapted_mu_local(k,m);
			std::vector<double> adapted_Pk_matrix(k,0.0);

			if ((num_subpop > 2) &&
				(gaussmix::gaussmix_adapt(S,num_subpop,sigma_vector,mu_local,Pk_matrix,
					adapted_sigma_vector,adapted_mu_local,adapted_Pk_matrix)) != 0)
			{
				// print results
				cout << "The matrix of adapted -1 subpop Pk's is " << endl;

				std::string s;
				for (int i = 0; i < k; i++)
				{
					s += adapted_Pk_matrix[i];
					s += " ";
				}
				cout << s << endl;


				cout << "The matrix of adapted -1 subpop mu's is " << endl;
				adapted_mu_local.print();

				for (int i = 0; i < k; i++)
				{
					cout << "The Covariance matrix " << i << " of adapted -1 subpop is " << endl;
					adapted_sigma_vector[i]->print();
				}

			}
			else
			{
				cout << "Error adapting to subpop -1" << endl;
			}

		    adapted_mu_local.clear();
		    adapted_Pk_matrix.clear();
			for (int i = 0; i < k; i++)
			{
				delete adapted_sigma_vector[i];
			}
		}
	}
	catch (...)
	{
		cout << "error encountered during processing " << endl;
	}


	for (int i = 0; i < k; i++)
	{
		delete sigma_vector[i];
	}
    return 0;
}
