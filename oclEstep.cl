#ifndef _OCLESTEP_CL
#define _OCLESTEP_CL

__kernel void
oclEstep(__global float* likelihood /* return value */, \
					const int n, \
					const int m, \
					const int k, \
					__global const float* X, \
					__global       float* p_nk_matrix, \
					__global const float* sigma_inverses, \
					__global const float* determinants, \
					__global const float* mu_matrix, \
					__global const float* Pk_vec )
{
	__local float local_p_nk[ ESTEP_K ];
	__local float local_LSEtmp[ ESTEP_K ];

	// NOTE everything is 1-D.  D is 1-indexed
	const int D = get_work_dim();

	// NOTE we have n*k globals and k-locals, so n-groups
	// our group id is the datapoint (i.e., our n-value to operate with)
	const int gIdx = get_group_id(D-1);
	// our local id is the cluster (i.e., our k-value to operate with)
	const int lIdx = get_local_id(D-1);

	float priv_diff[ ESTEP_M ]; // (x-mu)
	float priv_X[ ESTEP_M ]; // data point
	float priv_sigma_inv[ ESTEP_M * ESTEP_M ]; // cov this cluster
	float priv_mu[ ESTEP_M ]; // mean for this cluster
	float priv_Pk = NAN; // weights
	float priv_log_P_xn = NAN; // total log-likelihood


	// for each data dimension save a private copy of the data
	for(int i=0; i<ESTEP_M; ++i)
	{
		// copy data point local
		priv_X[i] = X[ gIdx*ESTEP_M + i ];

		// copy this group's mean local
		priv_mu[i] = mu_matrix[ lIdx*ESTEP_M + i ];

		// copy this group's cov local
		for(int j=0; j<ESTEP_M; ++j)
		{
			// lIdx*m*m = matrix index
			// m*i      = matrix row index (i.e., ith row)
			// j        = matrix col index (i.e., jth col)
			int idx = (lIdx * ESTEP_M * ESTEP_M) + (i * ESTEP_M + j);
			priv_sigma_inv[i*ESTEP_M+j] = sigma_inverses[ idx ];
		}
	}

	// copy the weight for this cluster
	priv_Pk = Pk_vec[ lIdx ];

	// calculate (x-mu)
	for(int i=0; i<ESTEP_M; ++i)
		priv_diff[i] = priv_X[i] - priv_mu[i];

	// calculate (x-mu)^T * inv(cov) * (x-mu)
	// TODO make is not efficient using OpenCL vectors (i.e., float4)
	float priv_multtmp[ ESTEP_M ];
	float priv_expterm = 0.0;
	for( int i=0; i<ESTEP_M; ++i )
	{
		// first, (x-mu)^T * inv(cov)
		priv_multtmp[i]  = 0.0;
		for( int j=0; j<ESTEP_M; ++j )
		{	// row vector * matrix column
			priv_multtmp[i] += dot( priv_diff[j], priv_sigma_inv[ i + j*ESTEP_M ] );
		}

		// then, * (x-mu)
		priv_expterm += dot( priv_multtmp[i], priv_diff[i] );
	}

	// calculate log( N(x|mu,sigma) )
	// N = -0.5 * [(x-mu)^T *inv(cov) *(x-mu)] + -0.5*{ m*log(2*pi) + log(det) }
//	float priv_logpdf = (- 0.5) * ( ESTEP_M * log(2 * M_PI) + log( determinants[lIdx] ) + priv_expterm );
	float priv_logpdf = mad( (- 0.5), priv_expterm, dot( (- 0.5), mad( ESTEP_M, log(2.0 * M_PI), log(determinants[lIdx]) ) ) );

	// calculate log( w * N(x|mu,sigma) )
	local_p_nk[ lIdx ] = log(priv_Pk) + priv_logpdf;
	barrier( CLK_LOCAL_MEM_FENCE );

	// find total likelihood with the log-sum-exp trick for (numerical stability)
	// find max log-likelihood value for this datapoint
	float max_p_nk = NAN;
	for( int i=0; i<ESTEP_K; ++i )
		max_p_nk = fmax( max_p_nk, local_p_nk[ i ] );
	
	// E portion of the LSE trick
	local_LSEtmp[lIdx] = exp( local_p_nk[lIdx] - max_p_nk );
	barrier( CLK_LOCAL_MEM_FENCE );
	
	// S portion of the LSE trick
	float tmpLikelihood = 0.0;
	for( int i=0; i<ESTEP_K; ++i )
		tmpLikelihood += local_LSEtmp[i]; 

	// L portion of the LSE trick
	priv_log_P_xn = log(tmpLikelihood) + max_p_nk;

	// apply normalization weight for this cluster
	local_p_nk[lIdx] -= priv_log_P_xn;
	barrier( CLK_LOCAL_MEM_FENCE );

	// save off log-likelihood value for this model at this data point (this's n=gIdx, k=lIdx)
	p_nk_matrix[ gIdx*ESTEP_K + lIdx ] = local_p_nk[lIdx];

	// save off our log likelihood if we're the first WI in this WG
	if( lIdx == 0 )
		likelihood[ gIdx ] = priv_log_P_xn;

	// NOTE: summation of likelihood buffer for final likelihood should be done in a separate kernel or by the host since we can only synchronize  within a WG
}
#endif /* _OCLESTEP_CL */
