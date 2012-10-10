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

#include <iostream>
#include <iostream>
#include <ostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <set>

#include <stdio.h>

#include "KMeans.h"

/*! \file KMeans.cpp
*   \brief implementations for kmeans clustering algorithm
*/


/*
 * SET THIS TO 1 FOR DEBUGGING STATEMENT SUPPORT (via std out)
 */
#define debug 0

//#define statements - change #debug to 1 if you want to see EM's calculations as it goes
#define sqr(x) ((x)*(x))

#define MAX_CLUSTERS 50

//MAX_ITERATIONS is the while loop limiter for Kmeans
#define MAX_ITERATIONS 100

using namespace std;

/********************************************************************************************************
 * 						INTERNAL FUNCTION PROTOTYPES
 ********************************************************************************************************/

void all_distances(int m, int n, int k, double *X, double *centroid, double *distance_out);
int assignment_change_count (int n, int a[], int b[]);
void calc_cluster_centroids(int m, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid);
double calc_total_distance(int m, int n, int k, double *X, double *centroids, int *cluster_assignment_index);
void choose_all_clusters_from_distances(int m, int n, int k, double *X, double *distance_array, int *cluster_assignment_index);
void cluster_diag(int m, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid);
void copy_assignment_array(int n, int *src, int *tgt);
double euclid_distance(int m, double *p1, double *p2);
void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int *cluster_member_count);

/*************************************************************************************************************
 *                                   SUPPORT FUNCTIONS
 **************************************************************************************************************/


/*! \brief all_distances calculates distances from the centroids you initialized to every data point.
*
* In order to determine which data point belongs to which cluster, you calculate the distance between each point to each cluster - whichever distance
* is the smallest from that sampling determines the cluster assignment.
*	input - dimensionality, number of data points, number of clusters, double*s containing your data, cluster centroids and the distance calculation
*	output - void
*/

void all_distances(int m, int n, int k, double *X, double *centroid, double *distance_out)
{
	//for each data point
	for (int ii = 0; ii < n; ii++)
	{
		//for each cluster
		for (int jj = 0; jj < k; jj++)
		{
			distance_out[ii*k + jj] = euclid_distance(m, &X[ii*m], &centroid[jj*m]);
		}
	}
}

/*! \brief assignment_change_count keeps track of how many cluster assignments have changed.
*/

int assignment_change_count (int n, int a[], int b[])
{
	int change_count = 0;
	for (int ii = 0; ii < n; ii++)
		if (a[ii] != b[ii])
			change_count++;
	return change_count;
}


/*! \brief calc_cluster_centroids is the function that actually recalculates the values for centroids based on their reassignment.
*
* This ensures that the cluster centroids are still the means of the data that belong to them. Here is also where the double* that
* holds the new cluster centroids is assigned and filled in.
*	input - data, assignment index
*	output - void
*/

void calc_cluster_centroids(int m, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid)
{
	//for each cluster
	for (int b = 0; b < k; b++)
		if (debug) printf("\n%f\n", new_cluster_centroid[b]);

	int cluster_member_count[k];

	// initialize cluster centroid coordinate sums to zero
	for (int ii = 0; ii < k; ii++)
	{
		for (int jj = 0; jj < m; jj++)
		{
			new_cluster_centroid[ii*m + jj] = 0;
		}
	}
	//for each data point
	for (int ii = 0; ii < n; ii++)
	{
		// which cluster it's in
		int active_cluster = cluster_assignment_index[ii];

		// sum point coordinates for finding centroid
		for (int jj = 0; jj < m; jj++)
			new_cluster_centroid[active_cluster*m + jj] += X[ii*m + jj];
	}
	// divide each coordinate sum by number of members to find mean(centroid) for each cluster
	for (int ii = 0; ii < k; ii++)
	{
		get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
		if (cluster_member_count[ii] == 0)
			cout << "Warning! Empty cluster. \n" << ii << endl;

		// for each dimension
		for (int jj = 0; jj < m; jj++)
			new_cluster_centroid[ii*m + jj] /= cluster_member_count[ii];
			// warning!! will divide by zero here for any empty clusters
	}
}


/*! \brief calc_total_distance computes the total distance from all their respective data points for all the clusters you initialized.
*
* This function also initializes the array that serves as the index of cluster assignments for each point (i.e. which cluster each point "belongs" to on this iteration).
*	input - double*s containing your data, initial centroids
*	output - double holding the total distances
* note: a point with a cluster assignment of -1 is ignored.
*/

double calc_total_distance(int m, int n, int k, double *X, double *centroids, int *cluster_assignment_index)
{
	double tot_D = 0;
	//for each data point
	for (int ii = 0; ii < n; ii++)
	{
		//which cluster it's in
		int active_cluster = cluster_assignment_index[ii];
		//sum distance
		if (active_cluster != -1)
			tot_D += euclid_distance(m, &X[ii*m], &centroids[active_cluster*m]);
	}
	return tot_D;
}


/*! \brief choose_all_clusters_from_distances is the function that reassigns clusters based on distance to data points.
*
* This is the piece that smooshes clusters around to keep minimizing the distance between clusters and their data.
*	input - data, the array that holds the distances, and the assignment index
*	output - void
*/

void choose_all_clusters_from_distances(int m, int n, int k, double *X, double *distance_array, int *cluster_assignment_index)
{
	//for each data point
	for (int ii = 0; ii < n; ii++)
	{
		int best_index = -1;
		double closest_distance = -1;

		//for each cluster
		for (int jj = 0; jj < k; jj++)
		{
			//distance between point and centroid
			double cur_distance = distance_array[ii*k + jj];
			if ((closest_distance < 0) || (cur_distance < closest_distance))
			{
				best_index = jj;
				closest_distance = cur_distance;
			}
		}
		// store in the array
		cluster_assignment_index[ii] = best_index;
	}
}

/*! \brief cluster_diag diagrams the current cluster member count and centroids and prints them out for the user after each iteration.
*
*	input - data, assignment index, centroids
*	output - void
*/

void cluster_diag(int m, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid)
{
	int cluster_member_count[MAX_CLUSTERS];
	//get the current cluster member count
	get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
	if (debug) cout << "  Final clusters \n" << endl;

	//print the current cluster centroids
	for (int ii = 0; ii < k; ii++)
	{
		if (debug) printf("cluster %d:  members: %8d\n", ii, cluster_member_count[ii]);
		printf(" ( ");
		for (int jj = 0; jj < m; jj++)
		{
			printf("%lf ",cluster_centroid[ii*m + jj]);
			if (jj != m-1)
				printf(", ");
		}
		printf(" ) \n\n ");

		fflush(stdout);
	}

	//print which data point belongs to which cluster
	if (debug)
	{
		cout << "member list" << endl;
		for (int ii = 0; ii < n; ii++)
		{
			printf(" %d, %d \n", ii, cluster_assignment_index[ii]);
		}
		cout << "--------------------------" << endl << flush;
	}
}


/*! \brief copy_assignment_array simply copies the assignment array (which point "belongs" to which cluster) so you can use it for the next iteration.
*
*	input - source and target
*	output - void
*/

void copy_assignment_array(int n, int *src, int *tgt)
{
	for (int ii = 0; ii < n; ii++)
		tgt[ii] = src[ii];
}

/*! \brief euclid_distance calculates the euclidean distance between two points.
*
* This is the method used to assign data points to clusters in kmeans; the aim is to assign each point to the "closest" cluster centroid.
*	input - the dimensionality of the data, and two double*s representing point 1 and point 2
*	output - double which stores the calculated distance
*/

double euclid_distance(int m, double *p1, double *p2)
{
	double distance_sum = 0;
	for (int ii = 0; ii < m; ii++)
		distance_sum += pow(p1[ii] - p2[ii],2);
	return sqrt(distance_sum);
	if (debug) cout << "this iteration's distance sum is " << distance_sum << endl;
}


/*! \brief get_cluster_member_count takes the newly computed cluster centroids and basically takes a survey of how many points belong to each cluster.
*
* This is where the int* representing the number of data points for every cluster is initialized and filled in.
*	input - assignment index
*	output - void
*/

void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int * cluster_member_count)
{
	// initialize cluster member counts
	for (int ii = 0; ii < k; ii++)
		cluster_member_count[ii] = 0;

	// count members of each cluster
	for (int ii = 0; ii < n; ii++)
		cluster_member_count[cluster_assignment_index[ii]]++;

}


/******************************************************************
 *    IMPLEMENTATIONS OF PUBLIC FUNCTIONS
 ******************************************************************/

double * gaussmix::kmeans(int m, double *X, int n, int k)
{
	//holds the computed cluster_centroids to pass to EM later
    	double *cluster_centroid = new double[m*k];

	//for each data point, the distance to each centroid
	double *dist = new double[n*k];
	if (debug) printf("%p \n",dist);

	//the current cluster assignment
	int *cluster_assignment_cur = new int[n];
	if (debug) printf("%p \n",cluster_assignment_cur);

	//the previous cluster assignment
	int *cluster_assignment_prev = new int[n];

	//this keeps track of how many points have moved around - necessary to determine convergence
	double *point_move_score = new double[n*k];

	if (!dist || !cluster_assignment_cur || !cluster_assignment_prev || !point_move_score || n < k)
	{
		cout << "Error allocating arrays. \n" << endl;
		return NULL;
	}

	// give the initial cluster centroids some values randomly drawn from your data set
    srand( time(NULL) );
    std::set<int> choices;
    for (int i = 0; i < k; i++)
	{

    	// randomly choose a row from those we haven't already chosen
    	int row = 0;
    	do
    	{
    		row = rand() % n;

    	} while (choices.find(row) != choices.end());

    	choices.insert(row);

    	if (debug) cout << "picked row: " << row << endl;

		for (int j = 0; j < m; j++)
		{
        		cluster_centroid[i*m + j] = X[row*m + j];

		}
	}

	//calculate distances
	all_distances(m, n, k, X, cluster_centroid, dist);

	//pick clusters from the previously calculated distances
	choose_all_clusters_from_distances(m, n, k, X, dist, cluster_assignment_cur);

	//copy current to previous
	copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

	// batch update
	double prev_totD = -1;

	int batch_iteration = 0;

	while (batch_iteration < MAX_ITERATIONS)
	{
		if (debug) printf("batch iteration %d \n", batch_iteration);

		//diagram the current cluster situation
		cluster_diag(m, n, k, X, cluster_assignment_cur, cluster_centroid);

		//calculate the cluster centroids
		calc_cluster_centroids(m, n, k, X, cluster_assignment_cur, cluster_centroid);

		//store the total distance calculated by calc_total_distance in a double for further use
		double totD = calc_total_distance(m, n, k, X, cluster_centroid, cluster_assignment_cur);

		//smoosh points around to nearest cluster by recalculating distances
		all_distances(m, n, k, X, cluster_centroid, dist);

		//pick new clusters based on new distance calculation
		choose_all_clusters_from_distances(m, n, k, X, dist, cluster_assignment_cur);

		//keep track of how many data points moved clusters
		int change_count = assignment_change_count(n, cluster_assignment_cur, cluster_assignment_prev);
		if (debug) printf("batch iteration:%3d  dimension:%u  change count:%9d  totD:%16.2f totD-prev_totD:%17.2f\n", batch_iteration, 1, change_count, totD, totD-prev_totD);

		//store totD as your previous totD
		prev_totD = totD;

		//increment through your total number of iterations
		batch_iteration++;

		// done with this phase if nothing has changed
		if (change_count == 0)
		{
			if (debug) cout << "No change made on this step - reached convergence. \n" << endl;
			break;
		}
		copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

	}
	//sanity check
	if (debug) cluster_diag(m, n, k, X, cluster_assignment_cur, cluster_centroid);

	delete[] dist;
	if (debug) printf("%p \n",cluster_assignment_cur);
	delete[] cluster_assignment_cur;
	delete[] cluster_assignment_prev;
	delete[] point_move_score;

	//return the final centroids calculated by Kmeans for use by EM later
    	return cluster_centroid;

}
