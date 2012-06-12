/* EXPECTATION MAXIMIZATION ALGORITHM
   Written by Elizabeth Garbee
   Cyberpoint International, LLC 2012 */

/* An Expectation Maximization algorithm computes the MLEs of latent variables and unknown parameters in probabilistic models. In each iteration, the E step computes the function of theta and theta_old which is a lower bound on the log probablility of X (observed data) and theta (log-likelihood). The M step maximizes (saturates) that lower bound. In the subsequent E step, the new bound is at least as high as the previous one. Thus EM monotonically increases the likelihood of X - as long as the log probablility of X and theta is less than infinity, EM necessarily converges.

This algorithm uses K means to seed the EM iteration, providing it with centroid clusters as initial guesses, and thus increasing the probablility that EM finds a global max. */

/* theta = log-likelihood
   theta_old = log-likelihood from previous iteration
   theta_new = log-likelihood from current iteration
   prob = probability
   mean = mean
   sd = standard deviation */

#include <stdio.h>

/* Picking starting parameters */

void start_em(int n, double * data, int k, double * prob, double * mean, double * sd)
{
	int i, j; double mean1 = 0.0, sd1 = 0.0;

	for (i = 0; i < n; i++)
		mean1 += data[i];
	mean1 /= n;
	for (i = 0; i < n; i++)
		sd1 += square(data[i] - mean1);
	sd1 = sqrt(sd1 / n);

	for (j = 0; j < k ; j++)
	{
		prob[j] = 1.0 / k;
		mean[j] = data[rand() % n];
		sd[j] = sd1;
	}
}

/* Update group memberships (which data belong to which group) */

void update_class_prob(int n, double * data, int k, double * prob, double * mean, double * sd, double ** class_prob)
{
	int i, j;

	for (i = 0; i < n; i++)
		for (j = 0; j < k; j++)
			class_prob[i][j] =
				classprob(j, data[i], k, prob, mean, sd);
}


/* Update mixture proportions */

void update_prob(int n, double * data, int k, double * prob, double ** class_prob)
{
	int i, j;
		
	for (int j = 0; j < k; j++)
		{
			prob[j] = 0.0;

			for (int i = 0; i < n; i++)
				prob[j] += class_prob[i][j];

			prob[j] /= n;
		}
}

/* Update component means (shift the means around in the clusters) */

void update_mean(int n, double * data, int k, double * prob, double * mean, double ** class_prob)
{
	int i, j;

	for (int j = 0; j < k; j++)
		{
			mean[j] = 0.0;
			
			for (int i = 0; i < n; i++)
				mean[j] += data[i] * class_prob[i][j];

			mean[j] /= n * prob[j] + TINY;
		}
}

/* Update component standard deviations */

void update_std(int n, double * data, int k, double * prob, double * mean, double * std, double ** class_prob)
{
	int i, j;
		
	for (int j = 0, j < k; j++)
	{
		sd[j] = 0.0;
	
		for (int i = 0; i < n; i++)
			sd[j] += square(data[i] - mean[j]) * class_prob[i][j];

		sd[j] /= (n * prob[j] + TINY);
		sd[j] = sqrt(sd[j]);
	}
}

/* Update the mixture */

void update_parameters(int n, double * data, int k, double_prob, double * mean, double * sd, double ** class_prob)
{
	/* update mixture proportions */
	update_prob(n, data, k, prob, class_prob);

	/* update mean for each component */
	update_mean(n, data, k, prob, mean, class_prob);

	/* update the standard deviation */
	update_sd(n, data, k, prob, mean, sd, class_prob);
}

/* The EM algorithm */

double em (int n, double * data, int k, double * prob, double * mean, double * sd, double eps)
{
	double theta = 0, theta_old = 0;
	double ** class_prob = alloc_matrix(n, k);

	start_em(n, data, k, prob, mean, sd);
	do 
	{
		theta_old = theta;
		update_class_prob(n, data, k, prob, mean, sd, class_prob);
		update_parameters(n, data, k, prob, mean, sd, class_prob);
		theta = mixTHETA(n, data, k, prob, mean, sd);
	}
	while ( !check_tol(theta, theta_old, eps) );

return theta;
}



