#include <iostream>

using namespace std;

// Kmeans Algorithm

int void double Compute(double[][] data, double threshold)
{
	int components = this.gaussians.Count;

	// create new

	KMeans kmeans = new KMeans(components);
	
	// compute the Kmeans

	kmeans.Compute(data, threshold);

	// initialize the mixture model with data from Kmeans

	NormalDistribution[] distributions = new NormalDistribution[components];
	double[] proportions = kmeans.Clusters.Proportions;
	for (int i = 0; i < components; i++)
	{
		double[] mean = kmeans.Clusters.Centroids[i];
		double[,] covariance = kmeans.Clusters.Covariances[i];
		distributions[i] = new NormalDistribution(mean, covariance);
	}

	// fit a multivariate Gaussian

	model = Mixture<NormalDistribution>.Estimate(data, threshold, proportions, distributions);

	// return the log-likelihood as a measure of goodness of fit

	return model.LogLikelihood(data);
}


// EM Algorithm

int void Fit(double[] observations, double[] weights, IFittingOptions options)
{
	// estimation parameters
	
	double threshold = 1e-3;
	IFittingOptions innerOptions = null;

	if (options != null)
	{
		// process optional arguments
		MixtureOptions o = (MixtureOptions)options;
		threshold = o.threshold
		innerOptions = o.Inneroptions
	}

	// initialize means, covariances and mixing coefficients
	// evaluate the initial value of the log-likelihood

	int N = observations.Length;
	int K = components.Length;

	double weightSum = weights.Sum

	// initialize responsibilites

	double[] norms = new double[N];
	double[][] gamma = new double[K][];
	for (int k = 0; k < gamma.Length; k++)
		gamma[k] = new double[N];

	// clone the current distribution values

	double[] pi = (double[])coefficients.Clone();
	T[]pdf = new T[components.Length];
	for (int i = 0; i < components.Length; i++)
		pdf[i] = (T)components[i].Clone();

	// prepare the iteration

	double likelihood = loglikelihood(pi, pdf, observations, weights);
	bool converged = false;

	// start

	while (!converged)
	{
		// expectation: evaluate the component distributions responsibilities using the current parameter 			values

		for (int k = 0; k < gamma.Length; k++)
			for (int i = 0; i < observations.Length; i++)
				norms[i] += gamma[k][i] = pi[k] * pdf[k].ProbabilityFunction(observations[i]);

		for (int k = 0; k < gamma.Length; k++)
			for (int i = 0; i < weights.Length; i++)
				if (norms[i] != 0) gamma [k][i] *= weights[i] / norms[i];

		// maximization: re-evaluate the distribution parameters using the previously computed 			responsibilities

		for (int k = 0; k < gamma.Length; k++)
		{
			double sum = gamma[k].Sum();

			for (int i = 0; i < gamma[k].Length; i++)
				gamma[k][i] /= sum;

			pi[k] = sum / weightSum;
			pdf[k].Fit(observations, gamma[k], innerOptions);
		}

		// evaluate the log-likelihood and check for convergence

		double newLikelihood - logLikelihood(pi, pdf, observations, weights);
		
		if (Double.IsNaN(newLikelihood) || Double.IsInfinity(newLikelihood))
			throw new ConvergenceException("Fit does not converge.");

		if(Math.Abs(likelihood - newLikelihood) < threshold * Math.Abs(likelihood))
			converged = true;

		likelihood = newLikelihood;
	}
	
	// become the newly fitted distribution function
 
	this.initialize(pi, pdf);
}
