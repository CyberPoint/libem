#ifndef EM_AGORITHM_H
#define EM_ALGORITHM_H

/* 
   EXPECTATION MAXIMIZATION ALGORITHM (header file)
   CYBERPOINT INTERNATIONAL, LLC
   Written by Elizabeth Garbee, Summer 2012

*/

#include <iostream>
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

/**************************************************************************
			SUPPORT FUNCTION DECLARATIONS
**************************************************************************/

/*ReadCSV is a function that simply reads in data from a comma delineated file and stores it in a string
for immediate use by ParseCSV.
	input - CSV data as a vector of strings
	output - void */

void ReadCSV(vector<string> &record, const string& line, char delimiter);

/* ParseCSV a function that parses the data you just read in from ReadCSV and returns a vector of doubles 
(which is the input format necessary for the kmeans function).
	input - string from ReadCSV
	output - vector of doubles containing your data */

vector<double> * ParseCSV(int argc, char *argv[]);

/* euclid_distance is a function that does just that - it calculates the euclidean distance between two points. This
is the method used to assign data points to clusters in kmeans; the aim is to assign each point to the "closest" cluster
centroid.
	input - two double*s representing point one and point 2
	output - double which stores the distance */

double euclid_distance(int dim, double *pl, double *p2);

/* all_distances calculates distances from the centroids you initialized in main to every data point.
	input - double*s containing your data, initial centroids
	output - void */

void all_distances(int dim, int n, int k, double *X, double *centroid, double *distance_out);

/* calc_total_distance computes the total distance from all their respective data points for all the clusters
you initialized. This function also initializes the array that serves as the index of cluster assignments 
for each point (i.e. which cluster each point "belongs" to on this iteration).
	input - double*s containing your data, initial centroids
	output - double */

double calc_total_distance(int dim, int n, int k, double *X, double *centroids, int *cluster_assignment_index);

/* choose_all_clusters_from_distances is the function that reassigns clusters based on distance to data points - this
is the piece that smooshes clusters around to keep minimizing the distance between clusters and their data.
	input - data, the array that holds the distances, and the assignment index
	output - void */

void choose_all_clusters_from_distances(int dim, int n, int k, double *X, double *distance_array, int *cluster_assignment_index);

/* calc_cluster_centroids is the function that actually recalculates the values for centroids based on their reassignment, in order
to ensure that the cluster centroids are still the means of the data that belong to them. This is also where the double* that
holds the new cluster centroids is assigned and filled in.
	input - data, assignment index
	output - void */

void calc_cluster_centroids(int dim, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid);

/* get_cluster_member_count takes the newly computed cluster centroids and basically takes a survey of how
many points belong to each cluster. This is where the int* representing the number of data points for 
every cluster is initialized and filled in.
	input - assignment index
	output - void */

void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int *cluster_member_count);

/* update_delta_score_table is the first step in reassigning points to the clusters that are now closest to them - it basically
creates a table of which clusters need to be moved and fills in that table. Not all points will need to be reassigned after
each iteration, so this function keeps track of the ones that do. 
	input - data, current cluster assignment, current cluster centroids, member count
	output - void */

void update_delta_score_table(int dim, int n, int k, double *X, int *cluster_assignment_cur, double *cluster_centroid, int *cluster_member_count, double*point_move_score_table, int cc);

/* perform_move is the piece that actually smooshes around the clusters. 
	input - data, cluster assignments, cluster centroids, and member counts
	output - void */

void perform_move (int dim, int n, int k, double *X, int *cluster_assignment, double *cluster_centroid, int *cluster_member_count, int move_point, int move_target_cluster);

/* cluster_diag gets the current cluster member count and centroids and prints them out for the user after each iteration.
	input - data, assignment index, centroids
	output - void */

void cluster_diag(int dim, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid);

/* copy_assignment_array simply copies the assignment array (which point "belongs" to which cluster)
so you can use it for the next iteration
	input - source and target
	output - void */ 

void copy_assignment_array(int n, int *src, int *tgt);

/* assignment_change_count keeps track of the count of how many points have been reassigned for each iteration.
	input - arrays a and b
	output - an integer representing how many points have "moved" (been reassigned) */

int assignment_change_count (int n, int a[], int b[]);

/***************************************************************************
		KMEANS AND EM DECLARATIONS
***************************************************************************/

void kmeans(int dim, double *X, int n, int k, double *cluster_centroid, int *cluster_assignment_final);






#endif //EM_ALGORITHM_HEADER

