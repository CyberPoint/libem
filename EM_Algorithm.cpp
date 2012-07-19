/*********************************************************************************************************
        EXPECTATION MAXIMIZATION ALGORITHM
	CYBERPOINT INTERNATIONAL, LLC
	Written by Elizabeth Garbee, Summer 2012
**********************************************************************************************************

Reference Numerical Recipes 3rd Edition: The Art of Scientific Computing 3 
Cambridge University Press New York, NY, USA ©2007 
ISBN:0521880688 9780521880688

Gaussian Mixture Models are some of the simplest examples of classification for unsupervised learning - they are also some of the simplest examples where a solution by an Expectation Maximization algorithm is very successful. Here's the setup: you have N data points in an M dimensional space, usually with 1 < M < a few. You want to fit this data (and if you don't, you might as well stop reading now), but in a special sense - find a set of K multivariate Gaussian distributions that best represents the observed distribution of data points. K is fixed in advance but the means and covariances are unknown. What makes this "unsupervised" learning is that you have "unknown" data, which in this case are the individual data points' cluster memberships. One of the desired outputs of this algorithm is for each data point n, an estimate of the probability that it came from distribution number k. Thus, given the data points, there are three parameters we're interested in approximating:

	mu - the K means, each a vector of length M
	sigma - the K covariance matrices, each of size M x M
	P - the K probabilities for each of N data points

We also get some fun stuff as by-products: the probability density of finding a data point at a certain position x, where x is the M dimensional position vector; and L denotes the overall likelihood of your estimated parameter set. L is actually the key to the whole problem, because you find the best values for the parameters by maximizing the likelihood of L. This particular implementation of EM actually first implements a Kmeans approximation to provide an initial guess for cluster centroids, in an attempt to improve the precision and efficiency of the EM approximation.

Here's the procedure:
	-run a kmeans approximation on your data to give yourself a good starting point for your cluster centroids
	-guess starting values for mu's, sigmas, and P(k)'s
	-repeat: an Estep to get new p's and a new L, then an Mstep to get new mu's, sigmas and P(k)'s
	-stop when L converges (i.e. when the value of L stops changing
*/

// brick o' header files - don't forget the algorithm-specific one, please
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

// Include Riva Borbley's matrix class (which lives in Vulcan) for matrix math
#include "~/Vulcan/vulcan/src/utils/Matrix.cpp"

#define sqr(x) ((x)*(x))
#define MAX_CLUSTERS 16
#define BIG_double (INFINITY)

using namespace std;

void ReadCSV(vector<string> &record, const string& line, char delimiter);
vector<double> ParseCSV(void);
void kmeans(int dim, double *X, int n, int k, double *cluster_centroid, int *cluster_assignment_final);
vector<double> * ParseCSV(int argc, char *argv[]);
double euclid_distance(int dim, double *pl, double *p2);
void all_distances(int dim, int n, int k, double *X, double *centroid, double *distance_out);
double calc_total_distance(int dim, int n, int k, double *X, double *centroids, int *cluster_assignment_index);
void choose_all_clusters_from_distances(int dim, int n, int k, double *X, double *distance_array, int *cluster_assignment_index);
void calc_cluster_centroids(int dim, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid);
void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int *cluster_member_count);
void update_delta_score_table(int dim, int n, int k, double *X, int *cluster_assignment_cur, double *cluster_centroid, int *cluster_member_count,       double*point_move_score_table, int cc);
void perform_move (int dim, int n, int k, double *X, int *cluster_assignment, double *cluster_centroid, int *cluster_member_count, int move_point, int move_target_cluster);
void cluster_diag(int dim, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid);
void copy_assignment_array(int n, int *src, int *tgt);
int assignment_change_count (int n, int a[], int b[]);


/*************************************************************************************************************/
/** MAIN **

In Main, we call the function that parses data from a comma delineated file, store that data in a double*, then
give values for initial cluster centroids and allocate space for final cluster assignments. Once those
preliminaries are out of the way, we execute a kmeans analysis followed by an EM approximation, returning 
the "best" set of parameters to describe your mixture of gaussians.

**************************************************************************************************************/
int main(int argc, char *argv[])
{
	int i;

	vector<double> csv_data = ParseCSV();
	int n = csv_data.size();
	printf("n: %d\n", N);
	double pre[23];
	double * stored_data = pre;
	for (i = 0; i < N; i++)
	{
		stored_data[i] = csv_data[i];
		printf("%f |", stored_data[i]);
	} 
	// this section is hard coded for now - values need to be changed for specific use
	int dim = 1;
	int k = 3;
	// number of centroids you want it to use
	double pre_c_c[3] = {0, 0, 0};
	double *cluster_centroid = pre_c_c;
	int pre_c_a_f[23];
	int *cluster_assignment_final = pre_c_a_f;

	// perfom a KMeans analysis to determine initial cluster means for the EM algorithm to use
	kmeans(dim, stored_data, n, k, cluster_centroid, cluster_assignment_final);
	//EM();
}

/*************************************************************************************************************/
/** SUPPORT FUNCTIONS **
**************************************************************************************************************/

/* 

ReadCSV is a function that simply reads in data from a comma delineated file and stores it in a string
for immediate use by ParseCSV.
	input - CSV data as a vector of strings
	output - void
*/

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

/* 

ParseCSV a function that parses the data you just read in from ReadCSV and returns a vector of doubles 
(which is the input format necessary for the kmeans function).
	input - string from ReadCSV
	output - vector of doubles containing your data
*/

vector<double> ParseCSV(void)
{
	// read and parse the CSV data
		cout << "This algorithm takes mixture data in a CSV. Make sure to replace test_data.txt with your file name. \n";
		vector<string> row;
		vector<double> data;
		
		int line_number = 0;
		string line;
		ifstream in("littledata.csv"); // changed from test_data.csv
		if (in.fail())
		{
			cout << "File not found" << endl;
		}
		while (getline(in, line) && in.good())
		{
			ReadCSV(row, line, ',');
			// only looking at row[0] now because of an assumption of 1D data
			// count through line_number
			const char* s = row[0].c_str();
			data.push_back(atof(s));
			// now the data is stored in a vector of doubles called data
			//for (int i = 0, leng = row.size(); i < leng; i++)
			//{
				//cout << row[i] << "\t";
			//}
			//cout << endl;
		}
		in.close();
		return data;
		
}

/*

euclid_distance is a function that does just that - it calculates the euclidean distance between two points. This
is the method used to assign data points to clusters in kmeans; the aim is to assign each point to the "closest" cluster
centroid.
	input - two double*s representing point one and point 2
	output - double which stores the distance

*/

double euclid_distance(int dim, double *p1, double *p2)
{	
	double distance_sq_sum = 0;
	for (int ii = 0; ii < dim; ii++)
		distance_sq_sum += sqr(p1[ii] - p2[ii]);
	return distance_sq_sum;
}
/*

all_distances calculates distances from the centroids you initialized in main to every data point.
	input - double*s containing your data, initial centroids
	output - void

*/
void all_distances(int dim, int n, int k, double *X, double *centroid, double *distance_out)
{
	for (int ii = 0; ii < n; ii++) // for each data point
		for (int jj = 0; jj < k; jj++) // for each cluster
		{
			distance_out[ii*k + jj] = euclid_distance(dim, &X[ii*dim], &centroid[jj*dim]);
		}
}

/*

calc_total_distance computes the total distance from all their respective data points for all the clusters
you initialized. This function also initializes the array that serves as the index of cluster assignments 
for each point (i.e. which cluster each point "belongs" to on this iteration).
	input - double*s containing your data, initial centroids
	output - double 

*/

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
/*

choose_all_clusters_from_distances is the function that reassigns clusters based on distance to data points - this
is the piece that smooshes clusters around to keep minimizing the distance between clusters and their data.
	input - data, the array that holds the distances, and the assignment index
	output - void

*/

void choose_all_clusters_from_distances(int dim, int n, int k, double *X, double *distance_array, int *cluster_assignment_index)
{
	for (int ii = 0; ii < n; ii++) // for each data point
	{
		int best_index = -1;
		double closest_distance = 100000;

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

/*

calc_cluster_centroids is the function that actually recalculates the values for centroids based on their reassignment, in order
to ensure that the cluster centroids are still the means of the data that belong to them. This is also where the double* that
holds the new cluster centroids is assigned and filled in.
	input - data, assignment index
	output - void

*/

void calc_cluster_centroids(int dim, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid)
{
	for (int b = 0; b < k; b++)
		printf("\n%f\n", new_cluster_centroid[b]);
	int carray[3];
	int * cluster_member_count = carray;
	// initialize cluster centroid coordinate sums to zero
	for (int ii = 0; ii < k; ii++)
	{
		// cluster_member_count[ii] = 0;
		// for (int jj = 0; jj < dim; jj++)
		new_cluster_centroid[dim*k] = 0;
	}
	// sum all points for every point
	for (int ii = 0; ii < n; ii++)
	{
		// which cluster it's in
		int active_cluster = cluster_assignment_index[ii];
		// update member count in that cluster
		// cluster_member_count[active_cluster]++;
		// sum point coordinates for finding centroid
		for (int jj = 0; jj < dim; jj++)
			new_cluster_centroid[active_cluster*dim + jj] += X[ii*dim + jj];
	}
	// divide each coordinate sum by number of members to find mean(centroid) for each cluster
	for (int ii = 0; ii < k; ii++)
	{
		get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
		if (cluster_member_count[ii] == 0)
			cout << "Warning! Empty cluster. \n" << ii << endl;
		// for each dimension
		for (int jj = 0; jj < dim; jj++)
			new_cluster_centroid[ii*dim + jj] /= cluster_member_count[ii];
			// warning!! will divide by zero here for any empty clusters
	}
}

/*

get_cluster_member_count takes the newly computed cluster centroids and basically takes a survey of how
many points belong to each cluster. This is where the int* representing the number of data points for 
every cluster is initialized and filled in.
	input - assignment index
	output - void
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

/*

update_delta_score_table is the first step in reassigning points to the clusters that are now closest to them - it basically
creates a table of which clusters need to be moved and fills in that table. Not all points will need to be reassigned after
each iteration, so this function keeps track of the ones that do. 
	input - data, current cluster assignment, current cluster centroids, member count
	output - void
*/

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

/*

perform_move is the piece that actually smooshes around the clusters. 
	input - data, cluster assignments, cluster centroids, and member counts
	output - void
*/

void perform_move (int dim, int n, int k, double *X, int *cluster_assignment, double *cluster_centroid, int *cluster_member_count, int move_point, int move_target_cluster)
{
	int cluster_old = cluster_assignment[move_point];
	int cluster_new = move_target_cluster;
	// update cluster assignment array
	cluster_assignment[move_point] = cluster_new;
	// update cluster count array
	cluster_member_count[cluster_old]--;
	cluster_member_count[cluster_new]++;

	if (cluster_member_count[cluster_old] <= 1)
		cout << "Warning! can't handle single-member clusters \n" << endl;
	// update centroid array
	for (int ii = 0; ii < dim; ii++)
	{	
		cluster_centroid[cluster_old*dim + ii] -= (X[move_point*dim + ii] - cluster_centroid[cluster_old*dim + ii]) / cluster_member_count[cluster_old];
		cluster_centroid[cluster_new*dim + ii] += (X[move_point*dim + ii] - cluster_centroid[cluster_new*dim + ii]) / cluster_member_count[cluster_new];
	}
}

/* 

cluster_diag gets the current cluster member count and centroids and prints them out for the user after each iteration.
	input - data, assignment index, centroids
	output - void

*/

void cluster_diag(int dim, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid)
{
	int cluster_member_count[MAX_CLUSTERS];
	get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
	cout << "  Final clusters \n" << endl;
	for (int ii = 0; ii < k; ii++)
		printf("   cluster %d:       members: %8d, centroid(%.1f) \n", ii, cluster_member_count[ii], cluster_centroid[ii*dim + 0]/*, cluster_centroid[ii*dim  + 1]*/);
} 

/*

copy_assignment_array simply copies the assignment array (which point "belongs" to which cluster)
so you can use it for the next iteration
	input - source and target
	output - void
*/ 

void copy_assignment_array(int n, int *src, int *tgt)
{
	for (int ii = 0; ii < n; ii++)
		tgt[ii] = src[ii];
}

/*

assignment_change_count keeps track of the count of how many points have been reassigned for each iteration.
	input - arrays a and b
	output - an integer representing how many points have "moved" (been reassigned)
*/
int assignment_change_count (int n, int a[], int b[])
{
	int change_count = 0;
	for (int ii = 0; ii < n; ii++)
		if (a[ii] != b[ii])
			change_count++;
	return change_count;
}

/* 

tensor_product computes the Kronecker tensor product - N x N matrix where the jth column is the column vector version of x 
with each entry scaled by the jth component of the row vector 
	input - scaling factor, 
	output -
*/

float[][] tensor_product(vector<double> x_n, vector<double> mu, int len)
{
	// this piece just multiplies the two (x_n - mu)'s together
	csv_data.size() = int m;
	int len = m;
	float [len][len];
	vector<double> one = x_n - mu;
	vector<double> two = x_n - mu;
	for (i = 0; i < len; i++)
	{
		for (j = 0; j < len; j++)
		{
			float[i][j] = one[i] * two[i];
		}
	}
}

/******************************************************************************************************************
** K MEANS **

Kmeans is a clustering algorithm which takes in a pile of data and separates it into clusters - it will keep shifting 
the means of these clusters ("cluster centroids") until they stop moving, i.e. when the algorithm converges. Like the
EM algorithm, it works in two steps:

	- assign each data point to the component whose mean it is closest to, by Euclidean distance
	- for all components, re-estimate the mean as the average of data points assigned to that component

With Kmeans, convergence is guaranteed: you can't get stuck in an infinite loop of shifting a point back and forth
between two centroids. This algorithm is relatively fast and converges rapidly. Its main advantage is that it can
easily reduce a huge pile of data to a much smaller number of "centers," which can then be used as the starting points
for more sophisticated methods (like Expectation Maximization).

*******************************************************************************************************************/
void kmeans(int dim, double *X, int n, int k, double *cluster_centroid, int *cluster_assignment_final)
// dim = dimension of data
// double *X = pointer to data
// int n = number of elements
// int k = number of clusters
// double *cluster centroid = initial cluster centroids
// int *cluster_assignment_final = output

{
	double *dist = (double *)malloc(sizeof(double) * n * k);
	int *cluster_assignment_cur = (int *)malloc(sizeof(int) * n);
	int *cluster_assignment_prev = (int *)malloc(sizeof(int) * n);
	double *point_move_score = (double *)malloc(sizeof(double) * n * k);

	if (!dist || !cluster_assignment_cur || !cluster_assignment_prev || !point_move_score)
		cout << "Error allocating arrays. \n" << endl;

			
	// give the initial cluster centroids some values
	double array [3] = {0, 5, 10};

	cluster_centroid = array;
	
	// initial setup
	all_distances(dim, n, k, X, cluster_centroid, dist);
	choose_all_clusters_from_distances(dim, n, k, X, dist, cluster_assignment_cur);
	copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

	// batch update
	double prev_totD = 10000.0;
	printf("1: \n%lf\n", prev_totD);
	int batch_iteration = 0;
	while (batch_iteration < I)
	{
		printf("batch iteration %d \n", batch_iteration);
		printf("2: \n%lf\n", prev_totD);
		
		cluster_diag(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
		printf("2.5: \n%lf\n", prev_totD);
		// update cluster centroids
		calc_cluster_centroids(dim, n, k, X, cluster_assignment_cur, cluster_centroid);

		// deal with empty clusters
		// see if we've failed to improve
		
		printf("3: \n%lf\n", prev_totD);

		double totD = calc_total_distance(dim, n, k, X, cluster_centroid, cluster_assignment_cur);
		printf("4: \n%lf\n", prev_totD);
		printf("totD: %lf, prev_totD: %lf\n", totD, prev_totD);
		if (totD > prev_totD)
			// failed to improve - this solution is worse than the last one
			{
				// go back to the old assignments
				copy_assignment_array(n, cluster_assignment_prev, cluster_assignment_cur);
				// recalculate centroids
				calc_cluster_centroids(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
				printf(" negative progress made on this step - iteration completed (%.2f) \n", totD-prev_totD);
				// done with this phase
				break;
			}
		// save previous step
		copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);
		// smoosh points around to nearest cluster
		all_distances(dim, n, k, X, cluster_centroid, dist);
		choose_all_clusters_from_distances(dim, n, k, dist, X, cluster_assignment_cur);

		int change_count = assignment_change_count(n, cluster_assignment_cur, cluster_assignment_prev);
		printf("batch iteration:%3d  dimension:%u  change count:%9d  totD:%16.2f totD-prev_totD:%17.2f\n", batch_iteration, 1, change_count, totD, totD-prev_totD);
		fflush(stdout);

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
	// clean up
	free(dist);
	free(cluster_assignment_cur);
	free(cluster_assignment_prev);
	free(point_move_score);
}

/*****************************************************************************************************************
** EM ALGORITHM **

(The EM algorithm owes its effectiveness to the Jensen inequality, which states that if you have a concave-down function,
and you want to interpolate between two points on that function, then the function(interpolation) >= interpolation(function).)

This algorithm's main function is find the parameters (theta) that maximize the likelihood of your set of data (usually a 
mixture of gaussians). It does this by calculating a likelihood L(theta), which is equal to the log of the probability of 
the data x given a set of parameters theta:

	L(theta) = lnP(x|theta)

For a more detailed explanation of the procedure for the algorithm, see the comment block at the top of this code.
For a detailed mathematical derivation of EM, see chapter 16.1 in Numerical Recipes 3 (citation above). 

	mu_k = the K number of means, each with a vector of length M
	sigma_k = the K number of covariance matrices, each of size M x M
	P(k) = the fraction of all data points at some position x, where x is the m-dimensional position vector
	p_nk = the K number of probabilities for each of the N data points
	n = total number of data points
	k = total number of multivariate Gaussians
	x = M dimensional position vector
	x_n = observed position
	data_point = an individual data point
	gaussian = a specific gaussian (distribution)
	weight = the probability of finding a point at position x_n
	density = multivariate gaussian density
			(actually the log of the densities)


*******************************************************************************************************************/
//need to put the signatures for these three functions either in the header or up at the top
double estep(int n, int k, &vector<double> csv_data, vector<double> mu, Matrix sigma, double current_P(k) )
{	
	//weight = sum over k of (multivariate gaussian density)(P(k));
	for (int gaussian = 1.; gaussian < k; gaussian++)
	{
		for (int data_point = 1.; data_point < n; data_point++)
		{
			//need an x here that represents the position vector
			vector<int> x = csv_data.size();
			// define sigma inverse
			Matrix sigma & inv() throw (SizeError, LapackError) = sigma_inverse;
			csv_data.size() = int m;
			//double density = -.5 * (x-mu)*sigma_inverse*(x-mu)-(m/2)log(2 pi)-.5*logdet(sigma) -> this is just a dot product

			weight = (density)(current_P(k));
		}
	}
	//likelihood = product over n of (weight);
	for (data_point = 1; data_point < n; data_point++)
	{
		double likelihood = 1.;
		likelihood *= weight;
	}
	return likelihood;
	
}

vector<double> mstep(int n, int k, double density, double p_nk, double current_P(k), double weight, double theta)
{
	//p_nk == P(k|n) = (multivariate gaussian density)(P(k)) / weight
	double p_nk = (density)(current_P(k)) / weight;
	int x_n;
	csv_data.at(x_n) = x_n;
	
	//vector<double> mu = sum over n of (p_nk)(x_n) / sum over n of p_nk;
	//double sigma = sum over n of (p_nk)(x_n - mu)*(x_n - mu) / sum over n of p_nk;
	//double P(k) = (1 / n)(sum over n of p_nk);
	for (data_point = 1.; data_point < n; data_point++)
	{
		for (int gaussian = 1.; gaussian < k; gaussian++)
		{
			double mu = (p_nk)(x_n) / p_nk;
		}
		//sigma = (p_nk)(x_n - mu)*(x_n - mu) / p_nk; -> here's where you use the tensor product
		tensor_product (vector<double> x_n, vector<double> mu, int len);
		double numerator = float[][] * p_nk;
		double denominator = p_nk;
			for (data_point = 1.; data_point < n; data_point++)
			{
				Matrix sigma (int m, int m) = numerator / denominator;
			}
		double P(k) = (1. / n)(p_nk);
	}
	vector<double> theta = (mu, sigma, P(k));
	return theta;
}


void EM(int n, int k, int epsilon, &double cluster_centroid) //don't forget to add signature to .h file
{
	int iterations;	
	int max_iterations = 100.; //arbitrary at this point
	int epsilon = 0.001;
	// use the kmeans cluster centroids as the starting mu's for EM
	double kmeans_mu = cluster_centroid; //syntax is probably wrong, but this is what i want to do
	// create the covariance matrix with dimensions M x M, where M is the length of the vectors which are the rows in the matrix
	csv_data.size() = int m;
	Matrix sigma (int m, int m);
	// fill in the covariance matrix initially - will approximate values later
	for (int filler = 1.;filler < m-1.;filler++)
	{
		double val = 1.;
		assign(double val, int i, int j);
	}
	// initialize P(k)s to 1 for starters
	vector<double> initial_P(k) = 1.;

	double old_likelihood;
	//new_likelihood = estep(initial_mu, initial_sigma, initial_P(k));
	double current_mu = kmeans_mu;
	Matrix current_sigma (int m, int m) = Matrix initial_sigma (int m, int m);
	vector<double> current_P(k) = vector<double> initial_P(k);
	while ((old_likelihood > epsilon) && ((new_likelihood - old_likelihood) > epsilon) && (iterations < max_iterations))
	{
		old_likelihood = new_likelihood;
		//double new_likelihood = estep(current_mu, current_sigma, current_P(k))
		//mstep(current_mu, current_sigma, current_P(k)) = (new_mu, new_sigma, new_P(k))
		//theta = (new_mu, new_sigma, new_P(k))
		
	}
	//cout << "The best parameters for this mixture of Gaussians after an Expectation Maximization analysis is   \n" << theta << endl;

}

