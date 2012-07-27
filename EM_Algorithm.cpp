/*********************************************************************************************************/
       /* EXPECTATION MAXIMIZATION ALGORITHM (library)
	CYBERPOINT INTERNATIONAL, LLC
	Written by Elizabeth Garbee, Summer 2012 */
/**********************************************************************************************************

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

// EM specific header - rest of the #includes and function declarations are in there
#include "EM_Algorithm.h"
// Include Riva Borbley's matrix class (which lives in Vulcan now)
#include "/home/egarbee/gmm/Matrix.h"

#define sqr(x) ((x)*(x))
#define MAX_CLUSTERS 16
#define BIG_double (INFINITY)
#define debug 1

using namespace std;

//dummy main for compiling purposes

int main(int argc, char *argv[])
{
	int i;

	vector<double> csv_data;
	csv_data = ParseCSV();
	int n = csv_data.size();
	
	printf("n: %d\n", n);
	double pre[50];
	double * stored_data = pre;
	for (i = 0; i < n; i++)
	{
		stored_data[i] = csv_data[i];
		printf("%f |", stored_data[i]);
	} 
	cout << "loaded correctly \n";
	// this section is hard coded to fit 'test_data.csv' for now - values need to be changed for specific use
	int dim = 1;
	int k = 3;
	//int m = csv_data.size();
	int m = 1;
		//N AND M NEED TO BE DIFFERENT - N IS NUMBER OF DATA POINTS, M IS THE DIMENSION OF THE DATA
	// number of centroids you want it to use
	//double pre_c_c[3] = {0, 0, 0};
	//double *cluster_centroid = pre_c_c;
	//int pre_c_a_f[50];
	//int *cluster_assignment_final = pre_c_a_f;

	// perfom a KMeans analysis to determine initial cluster means for the EM algorithm to use
	//kmeans(dim, double *X, int n, int k);

	// run the EM function, which itself calls estep and mstep, returning a best fit (best log likelihood approx.) for your data
	EM(dim, stored_data, n, k, m); 
}

/*************************************************************************************************************/
/** SUPPORT FUNCTIONS **
**************************************************************************************************************/
// For detailed descriptions of these functions (including input and output), see the header file

// a function that reads in data from a comma deliniated file
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

// a function that parses the data you just read in and returns a vector of doubles (which is the input format necessary for the kmeans function)
vector<double> ParseCSV(void)
{
	// read and parse the CSV data
		cout << "This algorithm takes mixture data in a CSV. Make sure to replace test_data.txt with your file name. \n";
		vector<string> row;
		vector<double> data;
		
		int line_number = 0;
		string line;
		ifstream in("test_data.csv"); // changed from test_data.csv
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

// define failure
/*void fail(char *str)
{
	printf(str);
	exit(-1);
} */

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

void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int * cluster_member_count)
{
	// initialize cluster member counts
	for (int ii = 0; ii < k; ii++)
		cluster_member_count[ii] = 0;
	// count members of each cluster
	for (int ii = 0; ii < n; ii++)
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

void cluster_diag(int dim, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid)
{
	int cluster_member_count[MAX_CLUSTERS];
	get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
	cout << "  Final clusters \n" << endl;
	for (int ii = 0; ii < k; ii++)
		printf("   cluster %d:       members: %8d, centroid(%.1f) \n", ii, cluster_member_count[ii], cluster_centroid[ii*dim + 0]/*, cluster_centroid[ii*dim  + 1]*/);
} 

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

/******************************************************************************************************************
** K MEANS **
*******************************************************************************************************************/
double * kmeans(int dim, double *X, int n, int k)
// dim = dimension of data
// double *X = pointer to data
// int n = number of elements
// int k = number of clusters
// double *cluster centroid = initial cluster centroids
// int *cluster_assignment_final = output

{
    	double *cluster_centroid = (double*)malloc(sizeof(double) * dim * k);
	double *dist = (double *)malloc(sizeof(double) * n * k);
	int *cluster_assignment_cur = (int *)malloc(sizeof(int) * n);
	int *cluster_assignment_prev = (int *)malloc(sizeof(int) * n);
	double *point_move_score = (double *)malloc(sizeof(double) * n * k);

	if (!dist || !cluster_assignment_cur || !cluster_assignment_prev || !point_move_score)
		cout << "Error allocating arrays. \n" << endl;

			
	// give the initial cluster centroids some values
    srand( time(NULL) );
    for (int i = 0; i < k; i++)
        cluster_centroid[i] = X[rand() % n];
	
	// initial setup
	all_distances(dim, n, k, X, cluster_centroid, dist);
	choose_all_clusters_from_distances(dim, n, k, X, dist, cluster_assignment_cur);
	copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

	// batch update
	double prev_totD = 10000.0;
	printf("1: \n%lf\n", prev_totD);
	int batch_iteration = 0;
	while (batch_iteration < 100.)
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
	// clean up
	free(dist);
	free(cluster_assignment_cur);
	free(cluster_assignment_prev);
	free(point_move_score);
    return cluster_centroid;
}



/*****************************************************************************************************************/
/** EM ALGORITHM **/
/*

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

e_output estep(int n, int m, int k, double *X, Matrix sigma_matrix, Matrix mu_matrix, Matrix Pk_matrix)
{
	Matrix p_nk_matrix(n,k);
	//Matrix likelihood(1,1);

	//Matrix denominator(1,1);
	//Matrix weight(1,n);

	Matrix data(X,n,m, Matrix::ROW_MAJOR);
	int pi = 3.141592653;
	int data_point;
	int gaussian;

	for (data_point = 1; data_point < n; data_point++)
	{
		cout << "beginning iteration 1 of n \n" << endl;
		Matrix x(1,m);
		for (int dim = 0; dim < m; dim++)
		{
			double data = x.getValue(data_point,m);
			x.assign(data,data_point,dim);
		}
		double weight_d;

		for (gaussian = 1; gaussian < k; gaussian++)
		{
			cout << "beginning iteration 1 of n \n" << endl;
			//please keep all hands and feet inside the vehicle at all times.
			
			//calculate the numerator = exp[-.5 * (x-mu) * sigma_inverse * (x-mu)]
			//difference = x-mu
			Matrix& difference = x.subtract(mu_matrix);
			//sigma inverse
			Matrix& sigma_inv = sigma_matrix.inv();

			//multiply the difference by the sigma inverse by the difference 
			Matrix &term1 = sigma_inv.dot(difference);

			Matrix &term2 = term1.dot(difference);
			//make term2 a double
			double term2_d = term2.getValue(0,0);
			//exp
			double numerator = exp(-.5 * term2_d);
			//calculate the denominator = (2pi)^(m/2)*sigma_determinant^.5
			double determinant = sigma_matrix.det();
			double term3 = pow(2*pi, m/2);
			double term4 = pow(determinant, .5);
			double denominator = term3 * term4;
			//calculate multivariate gaussian density = numerator / denominator
			double density = numerator/denominator;

			if (debug)
				cout << density << endl;

			//calculate weight = density * Pk
			Matrix density_m(1,1);
			density_m.assign(density, 1, 1);
			Matrix &weight = density_m.dot(Pk_matrix);
			weight_d = weight.getValue(0,0);

			if (debug)
				cout << weight_d << endl;

			//calculate p_nk = density * Pk / weight
			double p_nk = density * Pk_matrix.getValue(1, gaussian) / weight_d;
			p_nk_matrix.assign(p_nk,data_point,gaussian);

			if (debug)
				p_nk_matrix.print();
		}
		//calculate likelihood 
		double likelihood;
		likelihood *= weight_d;
		cout << "The likelihood for this iteration is    \n" << likelihood << endl;
	}

	e_output e_output_instance;
	return e_output_instance;
}


m_output mstep(int n, int m, int k, double *X, Matrix p_nk_matrix)
{
	Matrix sigma_matrix(m,m);
	Matrix mu_matrix(1,m);
	Matrix Pk_matrix(1,m);
	Matrix data(X,n,m, Matrix::ROW_MAJOR);

	for (int data_point = 1; data_point < n; data_point++)
	{
		Matrix x(1,m);
		for (int dim = 0; dim < m; dim++)
		{
			double data_d = data.getValue(1,1);
			x.assign(data_d,data_point,dim);
		}
		for (int gaussian = 1; gaussian < k; gaussian++)
		{
			//calculate mus
			double x_d = x.getValue(0,data_point);
			double p_nk_d = p_nk_matrix.getValue(data_point,gaussian);
			double mu = p_nk_d * x_d / p_nk_d;
			mu_matrix.assign(mu,0,data_point);

			if (debug)
				mu_matrix.print();

			//calculate sigmas
			double mu_d = mu_matrix.getValue(0,data_point);
			double subtract = x_d - mu_d;
			double product = subtract * subtract;
			double numerator = product * p_nk_d;
			double sigma = numerator / p_nk_d;
			sigma_matrix.assign(sigma,0,data_point);

			if (debug)	
				sigma_matrix.print();

			//calculate Pks
			double factor = 1/n;
			double Pk_d = factor * p_nk_d;
			Pk_matrix.assign(Pk_d,1,m);
			
			if (debug)
				Pk_matrix.print();

		}
	}

	m_output m_output_instance;
	return m_output_instance;
}

void EM(int dim, double *X, int n, int k, int m)
{
	int epsilon = .001;
	cout << "n is " << n;
	cout << "\nm is: " << m << endl;
	Matrix data(X,n,m, Matrix::ROW_MAJOR);
	cout << "matrix data made \n";
	
	//initialize the Pks
	Matrix initial_Pk(1,m);
	cout << "matrix initial_Pk made \n";
	//initialize sigma
	Matrix initial_sigma(m,m);
	cout << "matrix initial_sigma made \n";
	//initialize likelihoods
	double likelihood;
	double old_likelihood;
	
	//take the cluster centroids from kmeans as initial mus 
	double *kmeans_mu = kmeans(dim, X, n, k); 
	//make a matrix
	Matrix kmeans_mu_m(dim,m);
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			kmeans_mu_m.assign(*(kmeans_mu + (i*dim + j)),i,j);
		}
	}
	
    	//for (int i = 0;i < k; i++)
	//{
       		//printf("%f\n", &kmeans_mu[i]);
	//}

	m_output m_output_instance;
	e_output e_output_instance;

	int counter = 0;

	while (likelihood - old_likelihood > epsilon)
	{
		likelihood = old_likelihood;
		if (counter == 0)
			e_output_instance = estep(n, m, k, X, initial_sigma, kmeans_mu_m, initial_Pk);

		else if (counter != 0)

			e_output_instance = estep(n, m, k, X, m_output_instance.sigma_matrix, m_output_instance.mu_matrix, m_output_instance.Pk_matrix);

		m_output_instance = mstep(n, m, k, X, e_output_instance.p_nk_matrix);
	}

	likelihood = e_output_instance.likelihood;

	cout << "The likelihood approximated by the EM algorithm with the best fit is   \n" << likelihood << endl;
	//don't forget matrix cleanup!!!!
	
	
}

