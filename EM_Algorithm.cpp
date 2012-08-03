/*********************************************************************************************************/

       /* EXPECTATION MAXIMIZATION ALGORITHM (library)
	CYBERPOINT INTERNATIONAL, LLC
	Written by Elizabeth Garbee, Summer 2012 */

/**********************************************************************************************************/

// EM specific header - rest of the #includes and function declarations are in there
#include "EM_Algorithm.h"
// Include Riva Borbley's matrix class (which lives in Vulcan atm)
#include "/home/egarbee/gmm/Matrix.h"
#include <stdio.h>
#include <string.h>
#define sqr(x) ((x)*(x))
#define MAX_CLUSTERS 16
#define BIG_double (INFINITY)
//TODO: change debug to 0 when done
#define debug 1
#define MAX_LINE_SIZE 100000

using namespace std;


int main(int argc, char *argv[])
{
	int i, k, m, n;
	if (argc != 4) cout << " Usage: <exec_name> <data_file> <num_dimensions> <num_data_points> <num_clusters>" << endl;
	//TODO: cut out this main and put in the README as an example ... no mains in libraries
	int errno = 0;
	sscanf(argv[2],"d", &m);
	sscanf(argv[3],"d", &n);
	sscanf(argv[4],"d", &k);
	if (errno != 0) 
	{
		cout << "Invalid inputs" << endl;
	}

	//reading in and parsing the data
	double * data = new double[n*m];
	
	if (ParseCSV(argv[1], data, n, m) != 1)
	{
		return 0;
	}
	

	// create vectors that hold pointers to the EM result matrices
	vector<Matrix *> sigma_vector;
	for (i = 0; i < k; i++)
	{
		Matrix*p = new Matrix(m,m);
		sigma_vector.push_back(p);
	}
	//if (debug) cout << " final sigma matrices " << sigma_vector << endl;
	Matrix mu_local(k,m);
	if (debug) mu_local.print();
	Matrix Pk_matrix(1,k);
	if (debug) Pk_matrix.print();
	Matrix p_nk_matrix_local(n,k);

	// run the EM function
	EM(n, m, k, data, p_nk_matrix_local, sigma_vector, mu_local, Pk_matrix);

	// free the matrices you allocated
	for (i = 0; i < k; i++)
	{
		free(sigma_vector[i]);
	}
	delete [] data;
}

/*************************************************************************************************************/
/** SUPPORT FUNCTIONS **
**************************************************************************************************************/
// For detailed descriptions of these functions (including input and output), see the header file

// a function that reads in data from a comma delineated file
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

//return 1 for success, zero for error
int ParseCSV(char *file_name, double *data, int n, int m)
{
	char buffer[MAX_LINE_SIZE];
	FILE *f = fopen(file_name, "r");
	if (f = NULL) 
	{
		cout << "Could not open file" << endl;
		return 0;
	}
	memset(buffer, 0, MAX_LINE_SIZE);
	int row = 0;
	int cols = 0;
	while (fgets(buffer,MAX_LINE_SIZE,f) != NULL)
	{
		if (buffer[MAX_LINE_SIZE - 1] != 0) 
		{
			cout << "Max line size exceeded at zero relative line " << row << endl;
			return 0;
		}

		int errno = 0;
		char *ptok = strtok(buffer, ",");
		if (ptok) sscanf(ptok, "lf", &data[row*m]);
		if (errno != 0)
		{
			cout << "Could not convert data at index " << row << " and " << cols << endl;
			return 0;
		}
		for (cols = 1; cols < m; cols++)
		{
			sscanf(strtok(NULL, ","), "lf", &data[row*m + cols]);
			
			if (errno != 0) 
			{
				cout << "Could not convert data at index " << row << " and " << cols << endl;
				return 0;
			}
		}

		memset(buffer, 0, MAX_LINE_SIZE);
	}
	return 1;
}

// define failure
/*void fail(char *str)
{
	printf(str);
	exit(-1);
} */

// calculates the euclidean distance between points	
double euclid_distance(int m, double *pl, double *p2)
{	
	double distance_sq_sum = 0;
	for (int ii = 0; ii < m; ii++)
		distance_sq_sum += sqr(pl[ii] - p2[ii]);
	return distance_sq_sum;
}
// calculates all distances
void all_distances(int m, int n, int k, double *X, double *centroid, double *distance_out)
{
	for (int ii = 0; ii < n; ii++) // for each data point
		for (int jj = 0; jj < k; jj++) // for each cluster
		{
			distance_out[ii*k + jj] = euclid_distance(m, &X[ii*m], &centroid[jj*m]);
		}
}

// calculate total distance
double total_distance(int m, int n, int k, double *X, double *centroids, int *cluster_assignment_index)
// point with a cluster assignment of -1 is ignored
{
	double tot_D = 0;
	for (int ii = 0; ii < n; ii++) // for each data point
	{
		int active_cluster = cluster_assignment_index[ii]; // which cluster it's in
		// sum distance
		if (active_cluster != -1) 
			tot_D += euclid_distance(m, &X[ii*m], &centroids[active_cluster*m]);
	}
	return tot_D;
}

double calc_total_distance(int m, int n, int k, double *X, double *centroids, int *cluster_assignment_index)
{
	double tot_D = 0;
	for (int ii = 0; ii < n; ii++) // for each data point
	{
		int active_cluster = cluster_assignment_index[ii]; // which cluster it's in
		// sum distance
		if (active_cluster != -1) 
			tot_D += euclid_distance(m, &X[ii*m], &centroids[active_cluster*m]);
	}
	return tot_D;
}

void choose_all_clusters_from_distances(int m, int n, int k, double *X, double *distance_array, int *cluster_assignment_index)
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

void calc_cluster_centroids(int m, int n, int k, double *X, int *cluster_assignment_index, double *new_cluster_centroid)
{
	for (int b = 0; b < k; b++)
		printf("\n%f\n", new_cluster_centroid[b]);
	int carray[3];
	int * cluster_member_count = carray;
	// initialize cluster centroid coordinate sums to zero
	for (int ii = 0; ii < k; ii++)
	{
		// cluster_member_count[ii] = 0;
		// for (int jj = 0; jj < m; jj++)
		new_cluster_centroid[m*k] = 0;
	}
	// sum all points for every point
	for (int ii = 0; ii < n; ii++)
	{
		// which cluster it's in
		int active_cluster = cluster_assignment_index[ii];
		// update member count in that cluster
		// cluster_member_count[active_cluster]++;
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

void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int * cluster_member_count)
{
	// initialize cluster member counts
	for (int ii = 0; ii < k; ii++)
		cluster_member_count[ii] = 0;
	// count members of each cluster
	for (int ii = 0; ii < n; ii++)
		cluster_member_count[cluster_assignment_index[ii]]++;

}

void update_delta_score_table(int m, int n, int k, double *X, int *cluster_assignment_cur, double *cluster_centroid, int *cluster_member_count, double *point_move_score_table, int cc)
{
	// for each point both in and not in the cluster
	for (int ii = 0; ii < n; ii++)
	{
		double dist_sum = 0;
		for (int kk = 0; kk < m; kk++)
		{
			double axis_dist = X[ii*m + kk] - cluster_centroid[cc*m + kk];
			dist_sum += sqr(axis_dist);
		}
		double mult = ((double)cluster_member_count[cc] / (cluster_member_count[cc] + ((cluster_assignment_cur[ii]==cc) ? -1 : +1)));
		point_move_score_table[ii*m + cc] = dist_sum * mult;
	}
}

void perform_move (int m, int n, int k, double *X, int *cluster_assignment, double *cluster_centroid, int *cluster_member_count, int move_point, int move_target_cluster)
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
	for (int ii = 0; ii < m; ii++)
	{	
		cluster_centroid[cluster_old*m + ii] -= (X[move_point*m + ii] - cluster_centroid[cluster_old*m + ii]) / cluster_member_count[cluster_old];
		cluster_centroid[cluster_new*m + ii] += (X[move_point*m + ii] - cluster_centroid[cluster_new*m + ii]) / cluster_member_count[cluster_new];
	}
}

void cluster_diag(int m, int n, int k, double *X, int *cluster_assignment_index, double *cluster_centroid)
{
	int cluster_member_count[MAX_CLUSTERS];
	get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);
	cout << "  Final clusters \n" << endl;
	for (int ii = 0; ii < k; ii++)
		printf("   cluster %d:       members: %8d, centroid(%.1f) \n", ii, cluster_member_count[ii], cluster_centroid[ii*m + 0]/*, cluster_centroid[ii*m + 1]*/);
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

m = dimension of data
double *X = pointer to data
int n = number of elements
int k = number of clusters
double *cluster centroid = initial cluster centroids
int *cluster_assignment_final = output

*******************************************************************************************************************/
double * kmeans(int m, double *X, int n, int k)


{
    	double *cluster_centroid = (double*)malloc(sizeof(double) * m * k);
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
	all_distances(m, n, k, X, cluster_centroid, dist);
	choose_all_clusters_from_distances(m, n, k, X, dist, cluster_assignment_cur);
	copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

	// batch update
	double prev_totD = 10000.0;
	//printf("1: \n%lf\n", prev_totD);
	int batch_iteration = 0;
	while (batch_iteration < 100.)
	{
		printf("batch iteration %d \n", batch_iteration);
		//printf("2: \n%lf\n", prev_totD);
		
		cluster_diag(m, n, k, X, cluster_assignment_cur, cluster_centroid);
		//printf("2.5: \n%lf\n", prev_totD);
		// update cluster centroids
		calc_cluster_centroids(m, n, k, X, cluster_assignment_cur, cluster_centroid);

		// deal with empty clusters
		// see if we've failed to improve
		
		//printf("3: \n%lf\n", prev_totD);

		double totD = calc_total_distance(m, n, k, X, cluster_centroid, cluster_assignment_cur);
		//printf("4: \n%lf\n", prev_totD);
		//printf("totD: %lf, prev_totD: %lf\n", totD, prev_totD);
		if (totD > prev_totD)
			// failed to improve - this solution is worse than the last one
			{
				// go back to the old assignments
				copy_assignment_array(n, cluster_assignment_prev, cluster_assignment_cur);
				// recalculate centroids
				calc_cluster_centroids(m, n, k, X, cluster_assignment_cur, cluster_centroid);
				printf(" negative progress made on this step - iteration completed (%.2f) \n", totD-prev_totD);
				// done with this phase
				break;
			}
		// save previous step
		copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);
		// smoosh points around to nearest cluster
		all_distances(m, n, k, X, cluster_centroid, dist);
		choose_all_clusters_from_distances(m, n, k, dist, X, cluster_assignment_cur);

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
	// clean up later
	//free(dist);
	//free(cluster_assignment_cur);
	///free(cluster_assignment_prev);
	//free(point_move_score);
    return cluster_centroid;
}



/*****************************************************************************************************************/
/** EM ALGORITHM **

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

double estep(int n, int m, int k, double *X, vector<Matrix *> &sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix)
{
	Matrix p_nk_matrix(n,k);
	if (debug) cout << "p_nk matrix";
	if (debug) p_nk_matrix.print();
	
	double likelihood = 1;
	
	Matrix data(X,n,m,Matrix::ROW_MAJOR);
	if (debug) cout << "data";
	if (debug) data.print();

	int pi = 3.141592653;
	int data_point;
	int gaussian;

	for (data_point = 0; data_point < n; data_point++)
	{
		if (debug) cout << "1:beginning iteration " << data_point << " of " << n << endl;
		
		Matrix x(1,m);
		
		double P_xn = 0;

		for (int dim = 0; dim < m; dim++)
		{
			//double data = x.getValue(data_point,m)	

			if (debug) cout << "trying to assign data " << X[m*dim + data_point] << " to location " << dim << " by " << data_point << endl;
		
			x.assign(X[m*data_point + dim],0,dim);

		}
			
		double weight_d;
		double z_max = 0;
		double log_densities[k];

		for (gaussian = 0; gaussian < k; gaussian++)
		{
			Matrix mu_matrix_row(1,m);
			
			//if (debug) cout << "trying to assign mu " << mu_matrix.getValue(gaussian,dim) << " to location " << dim << " by " << data_point << endl;
			for (int dim = 0; dim < m; dim++)
			{
				double temp = mu_matrix.getValue(gaussian,dim);
				mu_matrix_row.assign(temp,0,dim);
			}
		
			
			
			if (debug) cout << "2:beginning iteration 1 of n \n" << endl;
			//please keep all hands and feet inside the vehicle at all times.
			
			//calculate the numerator = exp[-.5 * (x-mu) * sigma_inverse * (x-mu)]
			//difference = x-mu
			int currStep = 0;
			if (debug) cout << currStep << endl; currStep++;
			
			if (debug) cout << "x row count is " << x.rowCount() << endl;
			if (debug) cout << "x col count is " << x.colCount() << endl;
			if (debug) cout << "mu_matrix row count is " << mu_matrix.rowCount() << endl;
			if (debug) cout << "mu_matrix col count is " << mu_matrix.colCount() << endl;
			if (debug) cout << "data" << endl;
			if (debug) data.print();
			if (debug) cout << "x" << endl;
			if (debug) x.print();
			if (debug) cout << "mu matrix" << endl;
			if (debug) mu_matrix.print();
			Matrix& difference_row = x.subtract(mu_matrix_row);
			if (debug) difference_row.print();

			//sigma inverse
			if (debug) cout << currStep << endl; currStep++;
			Matrix sigma_inv;
			for (int i = 0; i < sigma_matrix.size(); i++)
			{
				sigma_inv = sigma_matrix[i]->inv();	
			}
			
			double determinant = 1;
			for (int j = 0; j < sigma_matrix.size(); j++)
			{
				determinant = sigma_matrix[j]->det();
			}
			Matrix difference_column(m,1);
			Matrix term1 = sigma_inv.dot(difference_column);
			//multiply the difference by the sigma inverse by the difference 
			if (debug) cout << "multiplication" << endl; 
			
			for (int i = 0; i < m; i++)
			{
				difference_column.assign(difference_row.getValue(0,i),i,0);
			}
	
			if (debug) cout << currStep << endl; currStep++;
			if (debug) cout << "yet another multiplication" << endl;
			Matrix &term2 = term1.dot(difference_row);

			if (debug) term2.print();
		
			if (debug) cout << currStep << endl; currStep++;
			double term2_d = term2.getValue(0,0);

			//exp
			double log_unnorm_density = (-.5 * term2_d);

			//calculate the denominator = (2pi)^(m/2)*sigma_determinant^.5

			
			double term3 = pow(2*pi, m/2);
			double term4 = pow(determinant, .5);
			double log_norm_factor = log(term3 * term4);

			//calculate multivariate gaussian density = numerator / denominator
			double log_density = log_unnorm_density - log_norm_factor;
			
			double current_z = log(Pk_matrix.getValue(0,gaussian)) + log_density;
			if (current_z > z_max) z_max = current_z;

			log_densities[gaussian] = log(Pk_matrix.getValue(0,gaussian)) + (log_density);
			
			//calculate p_nk = density * Pk / weight
			p_nk_matrix.assign(log_densities[gaussian],data_point,gaussian);

			if (debug) p_nk_matrix.print();
			
		} //end of gaussian for loop
		for(gaussian = 0; gaussian < k; gaussian++)
		{
			P_xn += exp(log_densities[gaussian] - z_max);
		}
		double log_P_xn = log(P_xn)+z_max;


		for (int gaussian = 0; gaussian < k; gaussian++)
		{
			// re-normalize per-gaussian point densities
			p_nk_matrix.assign(p_nk_matrix.getValue(data_point,gaussian)-log_P_xn,data_point,gaussian);
					
		}	
		
		likelihood *= weight_d;
		cout << "The likelihood for this iteration is " << likelihood << endl;
	}

	return likelihood;
}


void mstep(int n, int m, int k, double *X, Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix)
{
	Matrix data(X,n,m, Matrix::ROW_MAJOR);

	for (int gaussian = 0; gaussian < k; gaussian++)
	{	
		Matrix sigma_hat(m,m);
		Matrix mu_hat(1,m);
		Matrix Pk_hat(1,k);
		double norm_factor = 0;
		double x[m];
		for (int data_point = 0; data_point < n; data_point++)
		{
			
			for (int dim = 0; dim < m; dim++)
			{
				x[dim] = X[m*data_point + dim]*exp(p_nk_matrix.getValue(data_point,gaussian));
			}
			mu_hat.add(x, m, 0);
			//todo: see if we need to deal w/ underflow here
			norm_factor += exp(p_nk_matrix.getValue(data_point,gaussian));
		}
		for (int dim = 0; dim < m; dim++)
		{
			mu_hat.assign(mu_hat.getValue(0,dim)/norm_factor,0,dim);
		}
		for (int data_point = 0; data_point < n; data_point++)
		{

			Matrix x_m(m,1);
			x_m.add(x,m,0);
	
			Matrix difference_row = x_m.subtract(mu_hat);

			Matrix difference_column (m,1);

			for (int i = 0; i < m; i++)
			{
				difference_column.assign(difference_row.getValue(0,i),i,0);
			}
			
			//magic tensor product
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < m; j++)
				{
					sigma_hat.assign(difference_row.getValue(0,i) * difference_column.getValue(0,j),i,j);
				}
			}
		}

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				sigma_hat.assign(sigma_hat.getValue(i,j)/norm_factor,i,j);
			}
		}

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				Pk_hat.assign((1.0/n)*norm_factor,i,j);
			}
		}

		//assign sigma_hat to sigma_matrix[gaussian]

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				sigma_matrix[i*m + j]->assign(sigma_hat.getValue(i,j),i,j);
			}
		}
		
		//assign mu_hat to mu_matrix[gaussian]
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				mu_matrix.assign(mu_hat.getValue(i,j),i,j);
			}
		}

		//assign Pk_hat to Pk_matrix[gaussian]
		for (int i = 0; i < m; i++)
		{
			Pk_matrix.assign(Pk_hat.getValue(0,i),0,i);
		}

	}
}

void EM(int n, int m, int k, double *X, Matrix &p_nk_matrix, vector<Matrix*> &sigma_matrix, Matrix &mu_matrix, Matrix &Pks)

{
	double epsilon = .001;

	if (debug) cout << "n is " << n;
	if (debug) cout << "\nm is: " << m << endl;
	Matrix data(X,n,m, Matrix::ROW_MAJOR);
	if (debug) cout << "matrix data made \n";
	

	//initialize likelihoods with dummy fillers
	double new_likelihood = 1;	
	double old_likelihood = 0;
	
	//take the cluster centroids from kmeans as initial mus 
	double *kmeans_mu = kmeans(m, X, n, k);

	//initialize array of identity covariance matrices, 1 per k
	
	for(int gaussian = 0; gaussian < k; gaussian++)
	{
		for (int j = 0; j < m; j++)
		{
			sigma_matrix[gaussian]->assign(1.0,j,j);
		}
	}
	//initialize matrix of mus from kmeans the first time
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < m; j++)
		{
			mu_matrix.assign(kmeans_mu[i*m + j],i,j);
			if (debug) cout << "assigning, k is " << k << ", kmeans_mu[i] is " << kmeans_mu[i] << " at dimensions i (" << i << ") and j(" << j << ") \n";
			//kmeans_mu_m.assign(kmeans_mu[i], j, i);
		}
	}
	//initialize Pks
	Matrix Pk_matrix(n,k);
	for (int gaussian = 0; gaussian < k; gaussian++)
	{
		Pk_matrix.assign(1.0/k,0,gaussian);
	}

	//main loop of EM - this is where the magic happens
	while (new_likelihood - old_likelihood > epsilon)
	{
		new_likelihood = old_likelihood;
		if (debug) cout << "likelihood is " << new_likelihood << endl;

		estep(n, m, k, X, sigma_matrix, mu_matrix, Pk_matrix);
		mstep(n, m, k, X, p_nk_matrix, sigma_matrix, mu_matrix, Pk_matrix);
		
	}

}

