#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <istream>

#define I 1000
// define the number of iterations
#define sqr(x) ((x)*(x))
#define MAX_CLUSTERS 16
#define BIG_double (INFINITY)

using namespace std;

// function that reads and parses a comma (or tab) delineated file
void ReadCSV(vector<string> &record, const string& line, char delimiter);

/*************************************************************************************************************/
/** MAIN **
**************************************************************************************************************/
int main(int argc, char *argv[])
{
	// read and parse the CSV data
	cout << "This algorithm takes mixture data in a CSV. Make sure to replace test_data.txt with your file name. \n"; 
	vector<string> row;
	string line;
	ifstream in("test_data.txt");
	if (in.fail())
	{
		cout << "File not found" << endl;
		return 0;
	}
	while (getline(in, line) && in.good())
	{
		ReadCSV(row, line, ',');
		for (int i = 0, leng = row.size(); i < leng; i++)
		{
			cout << row[i] << "\t";
		}
		cout << endl;
	}
	in.close();
	return 0;
}

/*************************************************************************************************************/
/** FUNCTIONS **
**************************************************************************************************************/

void ReadCSV(vector<string> &record, const string& line, char delimiter)
{
	int linepos = 0;
	int inquotes = false;
	char c;
	int i;
	int linemax = line.length();
	string curstring;
	record.clear();

	while(line[linepos]!=0 && linepos < linemax)
	{
		c = line[linepos];

		if(!inquotes && curstring.length()==0 && c=='"')
		{
			inquotes = true;
		}
		else if(inquotes && c=='"')
		{
			if((linepos+1 < linemax) && (line[linepos+1]=='"'))
			{
				curstring.push_back(c);
				linepos++;
			}
			else
			{
				inquotes = false;
			}
		}
		else if(!inquotes && c==delimiter)
		{
			record.push_back(curstring);
			curstring='"';
		}
		else if(!inquotes && (c=='\r' || c=='\n'))
		{
			record.push_back(curstring);
			return;
		}
		else
		{
			curstring.push_back(c);
		}
		linepos++;
	}
	record.push_back(curstring);
	return;
}

/**********************************************************************************************************************************************
** K MEANS **
**********************************************************************************************************************************************/

// define failure
void fail(char *str)
{
	printf(str);
	exit(-1);
}

// calculates the euclidean distance between points	
double euclid_distance(int dim, double *pl, double *p2)
{	
	double distance_sq_sum = 0;
	for (int ii = 0; ii < dim; ii++)
		distance_sq_sum += sqr(pl[ii] - p2[ii]);
	return distance_sq_sum;
}
// calculates all distances
void all_distances(int dim, int n, int k, double *X, double *centroid, double *distance_out)
{
	for (int ii = 0; ii < n; ii++) // for each data point
		for (int jj = 0; jj < k; jj++) // for each cluster
		{
			distance_out[ii*k + jj] = euclid_distance(dim, &X[ii*dim], &centroid[jj*dim]);
		}
}

// calculate total distance
double total_distance(int dim, int n, int k, double *X, double *centroids, int *cluster_assignment_index)
// point with a cluster assignment of -1 is ignored
{
	double tot_D = 0;
	for (int ii = 0; ii < n; ii++) // for each data point
	{
		int active_cluster = cluster_assignment_index[ii]; // which cluster it's in
		// sum distance
		if (active_cluster != -1) 
			tot_D += euclid_distance(dim, &X[ii*dim], &centroids[active_cluster*dim]);
	}
	return tot_D;
}

double calc_total_distance(int dim, int n, int k, double *X, double *centroids, int *cluster_assignment_index)
{
	double tot_D = 0;
	for (int ii = 0; ii < n; ii++) // for each data point
	{
		int active_cluster = cluster_assignment_index[ii]; // which cluster it's in
		// sum distance
		if (active_cluster != -1) 
			tot_D += euclid_distance(dim, &X[ii*dim], &centroids[active_cluster*dim]);
	}
	return tot_D;
}

void choose_all_clusters_from_distances(int dim, int n, int k, double *X, double *distance_array, int *cluster_assignment_index)
{
	for (int ii = 0; ii < n; ii++) // for each data point
	{
		int best_index = -1;
		double closest_distance = BIG_double;

		// for each cluster
		for (int jj = 0; jj < k; jj++)
		{
			// distance between point and centroid
			double cur_distance = distance_array[ii*k + jj];
			if (cur_distance < closest_distance)
			{
				best_index = jj;
				closest_distance = cur_distance;
			}
		}
		// store in the array
		cluster_assignment_index[ii] = best_index;
	}
}

void calc_cluster_centroids(int dim, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid)
{
	int cluster_member_count[MAX_CLUSTERS];
	// initialize cluster centroid coordinate sums to zero
	for (int ii = 0; ii < k; ii++)
	{
		cluster_member_count = 0;
		for (int jj = 0; jj < dim; j++)
			new_cluster_centroid[ii*dim + jj] = 0;
	}
	// sum all points for every point
	for (int ii = 0; ii < k; ii++)
	{
		// which cluster it's in
		int active_cluster = cluster_assignment_index[ii];
		// update member count in that cluster
		cluster_member_count[active_cluster]++;
		// sum point coordinates for finding centroid
		for (int jj = 0; jj < dim; jj++)
			new_cluster_centroid[active_cluster*dim + jj] += X[ii*dim + jj];
	}
	// divide each coordinate sum by number of members to find mean(centroid) for each cluster
	for (int ii = 0; ii < k; ii++)
	{
		if (cluster_member_count[ii] == 0)
			cout << "Warning! Empty cluster. \n" << ii << endl;
		// for each dimension
		for (int jj = 0; jj < dim; jj++)
			new_cluster_centroid[ii*dim + jj] /= cluster_member_count[ii];
			// warning!! will divide by zero here for any empty clusters
	}
}

void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int *cluster_member_count)
{
	// initialize cluster member counts
	for (int ii = 0; ii < k; ii++)
		cluster_member_count[ii] = 0;
	// count members of each cluster
	for (int ii = 0; ii < k; ii++)
		cluster_member_count[cluster_assignment_index[ii]]++;

}

void update_delta_score_table(int dim, int n, int k, double *X, int *cluster_assignment_cur, double *cluster_centroid, int *cluster_member_count, double *point_move_score_table, int cc)
{
	// for each point both in and not in the cluster
	for (int ii = 0; ii < n; ii++)
	{
		double dist_sum = 0;
		for (int kk = 0; kk < dim; kk++)
		{
			double axis_dist = X[ii*dim + kk] - cluster_centroid[cc*dim + kk];
			dist_sum += sqr(axis_dist);
		}
		double mult = ((double)cluster_member_count[cc] / (cluster_member_count[cc] + ((cluster_assignment_cur[ii]==cc) ? -1 : +1)));
		point_move_score_table[ii*dim + cc] = dist_sum * mult;
	}
}

void perform_move (int dim, int n, int k, double *X, int *cluster_assignment_cur, double *cluster_centroid, int *cluster_member_count, int move_point, int move_target_cluster)
{
	int cluster_old = cluster_assignment[move_point];
	int cluster_new = move_target_cluster;
	// update cluster assignment array
	cluster_assignment[move_point] = cluster_new;
	// update cluster count array
	cluster_member_count[cluster_old]--;
	cluster_member_count[cluster_new]++;

	if (cluster_member_count[cluster_old] <= 1)
		cout << "Warning! can't handle single-member clusters \n" << endl
	// update centroid array
	for (int ii = 0; ii < dim; ii++)
	{	
		cluster_centroid[cluster_old*dim + ii] -= (X[move_point*dim + ii] - cluster_centroid[cluster_old*dim + ii]) / cluster_member_count[cluster_old];
		cluster_centroid[cluster_new*dim + ii] += (X[move_point*dim + ii] - cluster_centroid[cluster_new*dim + ii]) / cluster_member_count[cluster_new];
	}
}

/* void cluster_diag(int dim, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid)
{
	int cluster_member_count[MAX_CLUSTERS];
	get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
	cout << "  Final clusters \n" << endl;
	for (int ii = 0; ii < k; ii++)
		cout << "   cluster

} */

void copy_assignment_array(int n, int *src, int *tgt)
{
	for (int ii = 0; ii < n; ii++)
		tgt[ii] = src[ii];
}

int assignment_change_count (int n, int a[], int b[])
{
	int change_count = 0;
	for (int ii = 0; ii < n; ii++)
		if (a[ii] != b[ii])
			change_count++;
	return change_count;
}

void kmeans(int dim, double *X, int n, int k, double *cluster_centroid, int *cluster_assignment_final)
// dim = dimension of data
// double *X = pointer to data
// int n = number of elements
// int k = number of clusters
// double *cluster centroid = initial cluster centroids
// int *cluster_assignment_final = output

{
	double *dist = (double *) malloc(sizeof(double)) * n * k);
	int *cluster_assignment_cur = (int *) malloc(sizeof(int) * n);
	int *cluster_assingment_prev = (int *) malloc(sizeof(int) * n);
	double *point_move_score = (double *) malloc(sizeof(double) * n * k;

	if (!dist || !cluster_assignment_cur || !cluster_assignment_prev || !point_move_score)
		fail("Error allocating distribution arrays");
	
	// initial setup
	calc_all_distances(dim, n, k, X, cluster_centroid, dist);
	choose_all_clusters_from_distances(dim, n, k, dist, cluster_assignment_cur);
	copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

	// batch update
	double prev_totD = BIG_double;
	int batch_iteration = 0;
	while (batch_iteration < I)
	{
		cout << "batch iteration " << batch_iteration << endl;
		cluster_diag(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
		// update cluster centroids
		calc_cluster_centroids(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
		// deal with empty clusters
		// see if we've failed to improve
		double totD = calc_total_distance(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
		if (totD > prev_totD)
			// failed to improve - this solution is worse than the last one
			{
				// go back to the old assignments
				copy_assignment_array(n, cluster_assignment_prev, cluster_assignment_cur);
				// recalculate centroids
				calc_cluster_centroids(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
				cout << "Negative progress on this step - iteration complete. \n" << endl;	
				// done with this phase
				break;
			}
		// save previous step
		copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);
		// smoosh points around to nearest cluster
		calc_all_distances(dim, n, k, X, cluster_centroid, dist);
		choose_all_clusters_from_distances(dim, n, k, dist, cluster_assignment_cur);
		int change_count = assignment_change_count(n, cluster_assignment_cur, cluster_assignment_prev);
		cout << " " << batch_iteration << " " << change_count << " " << totD << " " << totD - prev_totD);
		flush(stdout)

		// done with this phase if nothing has changed
		if (change_count == 0)
		{
			cout << "No change made on this step - iteration complete. \n" << endl;
			break;
		}

		prev_totD = totD;
		batch_iteration++;
	}
	// write to output array
	copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_final);
	free(dist);
	free(cluster_assignment_cur);
	free(cluster_assignment_prev);
	free(point_move_score);
}

/*
InitKMeans
InitEM
Expect
Max

*/










































/*************************************************************************************

The aim of the approximation is as follows - given a collection of inner products, we want to estimate their distribution using a mixture of N weighted gaussians. We want to choose those gaussians that result in the closest fit to the true distribution of the inner products. This requires us to not only compute the means and variances of the N gaussians, but also to determine for each data point how responsible each of the gaussians was in generating it.

	Beta is the weight
	Mu is the mean
	Sigma is the covariance
	use Theta to represent the set of unknowns (beta, mu, sigma)

However, in order to estimate these parameters, we will also need to determine how responsible each of our gaussians was in generating each data point.

Supposing we have a data matrix U with K data vectors, each of which is D dimensional, we want to compute the matrix J, where each J i,j represents the 	probability that the ith data point was generated by the jth gaussian. We want to compute the value of theta which maximizes the log probability of U given theta. 

Given an initial estimate of theta (theta0), we repeatedly perform two steps until the estimates of theta converge.

************************************************************************************/


