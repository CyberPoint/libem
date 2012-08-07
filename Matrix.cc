#include <lapacke.h>
#include <stdio.h>
#include <vector>
#include "Matrix.h"



// return the index of M[i][j] in column major array format
int cmIndex(int i, int j, int dim)
{
	return j*dim + i;
}


/**
create an empty matrix
*/
Matrix::Matrix()
{ 
	//columns = new std::vector< std::vector<double> >();
	entries = new double[0];
	changed = true;
	numRows=0;
	numCols=0;
}

/**
create matrix of zeroes, n rows, m columns
*/
Matrix::Matrix(int n, int m)
{
	numRows=n;
	numCols=m;
	entries = new double[n*m];
	for (int i=0; i<n*m; i++)
		entries[i] = 0;
	changed = false;
	for (int j=0; j<m; j++)
	{
		printf("initing %d th column \n", j);
		fflush(stdout);
		std::vector<double> temp;
		columns.push_back(temp);

		for (int i=0; i<n; i++)
		{
			printf("initing %d th row \n", i);
			fflush(stdout);

			columns[j].push_back(0);
		}
	}
	
}

/**
* Constructor to initialize matrix to 0's off-diagonal, and  a given vector on the diagonal (a la numpy.diag)
@param di -- vector of length len, to be used as the diagonal
@param len length of di
*/
Matrix::Matrix(double di[], int len)
{
	entries = new double[len*len];
	changed = false;
	int index=0;
	for (int i=0; i<len; i++)
	{
		columns.push_back(std::vector<double>());
	}
	for (int i=0; i< len; i++)
	{
		for (int j=0; j<len; j++)
		{
			if (i==j) 
			{
				columns[j].insert(columns[j].begin() + i, di[i]);
				entries[index++] = di[i];
			}
			else 
			{
				columns[j].insert(columns[j].begin() +i, 0);
				entries[index++] = 0;
			}
		}
	}
	numRows = len;
	numCols = len;
}

/**
Create a matrix from its column-major or row-major matrix representation
@param a array of row-major or column-major representation of matrix
@param nRows current number of rows
@param nCols current number of columns
@param major Orientation.ROW_MAJOR or Orientation.COLUMN_MAJOR
*/
Matrix::Matrix(double a[], int nRows, int nCols, Orientation major=COLUMN_MAJOR)
{
	numRows = nRows;
	numCols = nCols;
	changed = false;
	for (int i=0; i<numCols; i++)
	{
		columns.push_back(std::vector<double>());
	}
	if (major==COLUMN_MAJOR)
	{
		entries = new double[numRows*numCols];
		for (int k=0; k< numRows*numCols; k++)
		{
			entries[k] = a[k];
			changed = false;
			int j = k/numRows;
			int i = k%numRows;
			columns[j].insert(columns[j].begin() + i, a[k]);
		}
	}
	else if (major==ROW_MAJOR)
	{
		entries = new double[0]; // will fill in later when needed
		for (int k=0; k< numRows*numCols; k++)
		{
			changed = true;
			int i = k/numCols;
			int j = k%numCols;
			columns[j].insert(columns[j].begin() + i, a[k]);
		}
	}

}

/**
//TODO: this might be a bad idea -- once a reference is returned, user can change value whenever they want, but entries won't get updated.
Overload [] so that i-j th entry can be accessed and assigned by matrix[i, j]
@param i row number
@param j column number
@return reference to value in ith row, jth column
*/
/*double & Matrix::operator()(int i, int j) 
{
	changed = true; //unfortunately, we can't know whether user will changed the value, so have to assume he/she did
	return columns[j][i]; 
}*/


/**
@return the element in ith row, jth column (indexed from 0)
*/
double Matrix::getValue(int i, int j)
{
	std::vector<double> col = columns[j];
	return col[i];
}

/**
How many rows are in the matrix?
@return the number of rows
*/
int Matrix::rowCount()
{
	return numRows;
}

/**
How many columns are in the matrix?
@return the number of columns
*/
int Matrix::colCount()
{
	return numCols;
}

/**
Assign val to row i, column j
*/
void Matrix::assign(double val, int i, int j)
{
	if (j>=numCols || j<0) throw SizeError((char *)"Error: attempt to assign value to a non-existent column");
	if (i>=numRows || i<0) throw SizeError((char *)"Error: attempt to assign value to a non-existent row");
	columns[j].insert(columns[j].begin() + i, val);
	changed = true;
}

/**Overwrite val in the ith row, jth column of the matrix
	@param val double value to write
	@param i row number
	@param j column number*/
void Matrix::update(double val, int i, int j)
{
	if (j>=numCols || j<0) throw SizeError((char *)"Error: attempt to write value to a non-existent column");
	if (i>=numRows || i<0) throw SizeError((char *)"Error: attempt to write value to a non-existent row");
	columns[j][i] = val;
	changed = true;
}





/**
Insert row as the rowNum'th row in the matrix (indexed from 0)
@param row the row to insert 
@param rowSize the length of row (should be the same as colCount())
@param rowNum location at which to insert row -- row will be the rowNum'th row in the matrix (indexed from 0) (0 <= rowNum <= rowCount())
*/
Matrix & Matrix::insertRow(double row[], int rowSize, int rowNum) throw (SizeError)
{
	if (rowNum>numRows)
	{
		throw SizeError((char *)"Error: inserting a row without inserting prior rows is not supported."); // shouldn't be skipping rows
	}
	if (rowNum<0)
	{
		throw SizeError((char *)"Error: inserting a row without inserting successive rows is not supported.");
	}
	if ((numCols > 0) && (rowSize!=numCols))
	{
		throw SizeError((char *)"Error: attempted to insert row whose size does not match current number of columns in matrix.");
	}

	if (numCols == 0)
	{
		// need to first create empty column entries
		for (int i = 0; i < rowSize; i++)
		{
			columns.push_back(std::vector<double>());
		}
		numCols = rowSize;
	}
	for (int i=0; i<rowSize; i++)
	{

		std::vector<double> & col = columns[i];
		std::vector<double>::iterator iter = col.begin();
		col.insert(iter+rowNum, row[i]);
	}
	numRows++;
	changed = true;
	
}
	
/**
Insert col as the colNum'th column in the matrix (indexed from 0)
@param col the column to insert
@param colSize the length of col (should be the same as numRows())
@param colNum location at which to insert col -- col will be the colNum'th column in the matrix (indexed from 0)
(0  <= colNum <= colCount())
*/
Matrix & Matrix::insertColumn(double col[], int colSize, int colNum) throw (SizeError)
{
	if (colNum>numCols)
	{
		throw SizeError((char *)"Error: inserting a column without inserting prior columns is not supported."); // shouldn't be skipping cols
	}
	if (colNum<0)
	{
		throw SizeError((char *)"Error: inserting a column without inserting successive columns is not supported.");
	}
	if ((numRows > 0) && (colSize!=numRows))
	{
		throw SizeError((char *)"Error: attempted to insert a column whose size doesn't match current number of rows in matrix");
	}

	if (numRows==0) numRows = colSize;
	std::vector<double> newCol(col, col+colSize);
	columns.insert(columns.begin() + colNum, newCol);
	
	numCols++;
	changed = true;
}

/**
return a copy of rowOffset'th row of the matrix
@param rowOffset number of the row to retrieve (indexed from 0)
@return vector representation of the specified row
*/
std::vector<double> Matrix::getCopyOfRow(int rowOffset) throw (SizeError)
{
	if (rowOffset<0 || rowOffset > numRows)
		throw SizeError((char*)"Error: attempted to get copy of non-existent row");
	std::vector<double> row;
	for (int j=0; j<numCols; j++)
		row.push_back(columns[j][rowOffset]);
	return row;
}

/**
return a copy of colOffset'th row of the matrix
@param colOffset number of the column to retrieve (indexed from 0)
@return vector representation of the specified column
*/
std::vector<double> Matrix::getCopyOfColumn(int colOffset) throw (SizeError)
{
	if (colOffset<0 || colOffset > numCols)
		throw SizeError((char*)"Error: attempted to get copy of non-existent column");
	std::vector<double> col;
	for (int i=0; i<numRows; i++)
	{
		col.push_back(columns[colOffset][i]);
		//printf("%f ", columns[colOffset][i]);
	}
	fflush(stdout);
	return col;
}


// this is only used as a dummy function for LAPACK
lapack_logical leq(const double* a, const double* b)
{
	if (*a<=*b) return 1;
	else return 0;
}


/**
* 
@return the determinant. Note: only works for square matrices
*/
double Matrix::det() throw (LapackError, SizeError)
{
	if (numRows!=numCols) throw SizeError((char *)"Attempt to compute determinant of a non-square matrix");
	updateArray(); // update column-major array representation of the matrix, to include any recent changes
	char JOBVS = 'N'; // don't need Schur vectors
	char SORT ='N'; // don't need eigenvalues ordered
	LAPACK_D_SELECT2 SELECT= &leq;	 // not referenced if SORT == 'N'
	lapack_int N=numRows; 
	// make a copy of entries, as LAPACK will overwrite
	double* A = new double[N*N];
	for (int i=0; i<N*N; i++)
	{
		A[i] = entries[i];
	}
	int LDA=N;
	lapack_int SDIM; // output
	double *WR, *WI, *VS; // output (VS should not be referenced if JOBVS =='N')
	lapack_int LDVS=1; // leading dimension for VS; not used

	WR= new double[N]; // will be filled with the real part of eigenvalues
	WI= new double[N]; // will be filled with the imaginary part of eigenvalues
	VS = new double[N];// shouldn't have to do this


	// get the eigenvalues from LAPACKE
	int code = LAPACKE_dgees(LAPACK_COL_MAJOR, JOBVS, SORT, SELECT, N, A, LDA, &SDIM, WR, WI, VS, LDVS);
	if (code!=0) {throw LapackError((char*)"failed to get eigenvalues through LAPACKE_dgees");}

	// now WR holds the real portion of our eigenvalues, WI the imaginary portion

	// multiply the eigenvalues
	double prodR=WR[0]; //real part
	double prodI=WI[0]; // imaginary part 
	for (int i=1; i<N; i++)
	{
		double re = prodR*WR[i] - prodI*WI[i];
		double im = prodR*WI[i] + prodI*WR[i];
		prodR =re;
		prodI =im;
	}
	// clean up
	delete[] A;
	delete[] WR;
	delete[] WI;
	delete[] VS;
	return prodR; // prodI will always be 0 (complex eigenvalues will be complex conjugates)
}



/**
Matrix multiplication this*B
@param B matrix to multiply by
@return product of matrix multiplication: this*B
*/
Matrix & Matrix::dot(Matrix& B)
{
	
	if (numCols!=B.rowCount()) throw SizeError((char*)"Error: Attempted to multiply matrices with mismatched sizes");
	int cols1 = numCols;
	int rows2 = cols1;
	int rows1 = numRows;
	int cols2 = B.colCount();
	double* result = new double[rows1*cols2];
	for (int i=0; i<rows1; i++)
	{
		for (int j=0; j<cols2; j++)
		{
			double sum=0;
			for (int k=0; k<cols1; k++)
			{
				sum+=getValue(i, k)*B.getValue(k, j);
			}	
			result[cmIndex(i, j, rows1)] = sum;
		}
	}
	Matrix& R = *(new Matrix(result, rows1, cols2));
	delete[] result;
	return R;
}

/** Subtract Matrix B from this Matrix 
	@param B Matrix to subtract from this*/
Matrix& Matrix::subtract(Matrix& B)
{
	if ( (numRows!=B.rowCount())|| (numCols!=B.colCount()) ) 
		throw SizeError((char*)"Error: Attempted to subtract matrices with mismatched sizes");
	double * result = new double[numRows*numCols];
	for (int i=0; i<numRows; i++)
		for (int j=0; j<numCols; j++)
			result[cmIndex(i, j, numRows)] = getValue(i,j) - B.getValue(i,j);
	Matrix& R = *(new Matrix(result, numRows, numCols));
	delete[] result;
	return R;
}


/** 
matrix multiplication for matrices already represented in column-major form
*/
void Matrix::dot(double A[], double B[], int rows1, int cols1, int cols2, double result[])
{
	for (int i=0; i<rows1; i++)
	{
		for (int j=0; j<cols2; j++)
		{
			double sum=0;
			for (int k=0; k<cols1; k++)
			{
				sum+=A[cmIndex(i, k, rows1)]*B[cmIndex(k, j, cols1)];
			}	
			result[cmIndex(i, j, rows1)] = sum;
		}
	}
}



/**
*
Compute the covariance of a positive, definite matrix (numpy.covar)
@return the covariance matrix
*/
Matrix& Matrix::covar()
{
	double* ones = new double[numCols*numCols];
	for (int i=0; i< numCols*numCols; i++)
		ones[i] =1.0/numCols;
	
	double* matrixArrayT = new double[numRows*numCols]; // transpose of Matrix
	for (int i=0; i<numCols; i++)
	{
		for (int j=0; j<numRows; j++)
		{
			matrixArrayT[cmIndex(i, j, numCols)] = getValue(j, i);
		}
	}
	double* m = new double[numCols*numRows];

	dot(ones, matrixArrayT,  numCols, numCols, numRows, m);

	for (int i=0; i< numRows*numCols; i++)
	{
		m[i] = matrixArrayT[i] - m[i];
	}
	delete[] ones;
	delete[] matrixArrayT;
	// m is now the deviation matrix
	//compute m'*m /n
	double* mTrans = new double[numCols*numRows];
	for (int i=0; i<numCols; i++)
	{
		for (int j=0; j<numRows; j++)
		{
			mTrans[cmIndex(i, j, numCols)] = m[cmIndex(j, i, numRows)];
		}
	}
	double* result = new double[numRows*numRows];
	dot(mTrans, m, numRows, numCols, numRows, result);
	for (int i=0; i<numRows*numRows; i++)
	{
		result[i] = result[i]/(numRows-1);
	}
	
	delete[] m;
	delete[] mTrans;
	Matrix& ret = *(new Matrix(result, numRows, numRows));
	delete[] result;
	return ret;
}

/**
Invert the matrix
@return the inverse of this matrix
*/
Matrix& Matrix::inv() throw (SizeError, LapackError)
{
	if (numRows!=numCols) throw SizeError((char *)"Error: tried to invert a non-square matrix");
	int dim = numRows;
	updateArray();
	double* result = new double[dim*dim];
	// copy matrixArray into result, as it's going to get overwritten 
	for (int i=0; i<dim*dim; i++)
	{
		result[i] = entries[i];
	}
	lapack_int lda=dim;
	lapack_int* ipiv = new lapack_int[dim];
	// put result in LU form
	lapack_int code = LAPACKE_dgetrf(LAPACK_COL_MAJOR, dim, dim, result, lda, ipiv );
	if (code!=0)
	{
		throw LapackError((char*)"ERROR in LU factorization in Matrix::inv"); 
	}
	// use LU form to find inverse, put in result
	code = LAPACKE_dgetri(LAPACK_COL_MAJOR, dim, result, lda, ipiv);
	if (code!=0)
	{
		throw LapackError((char*)"Error in inversion in Matrix::inv"); 
	}
	
	delete[] ipiv;
	Matrix& R = *(new Matrix(result, dim, dim));
	delete[] result;
	return R;
}

/**
Add vector to matrix row by row or column by column (in place)
@param vector array to add
@param m length of vector
@param axis row by row (0) or column by column (1)
*/
void Matrix::add(double vector[], int m, int axis)
{
	if (axis==0)
	{
		if (numCols!=m){throw SizeError((char*)"Error in Matrix::add: row size doesn't match number of columns in matrix.");}
		for (int i=0; i<numRows; i++) // ith row
		{
			for (int j=0; j<numCols; j++) // jth column
			{
				columns[j][i] = columns[j][i] + vector[j];
			}
		}
	}
	else if (axis==1)
	{
		if (numRows!=m) {throw SizeError((char*)"Error in Matrix::add: column size doesn't match number of rows in matrix");}
		for (int j=0; j<numCols; j++) // jth column
		{	
			for (int i=0; i<numRows; i++) // ith row
			{
				columns[j][i] = columns[j][i] + vector[i]; 
			}
		}
	}
}

/**
Subtract vector from matrixArray row by row or column by column
@param vector array to subtract
@param m length of vector
@param axis row by row (0) or column by column (1)

*/
void Matrix::subtract(double vector[], int m, int axis)
{
	// call add with negated vector

	double* neg = new double[m];
	for (int i=0; i<m;i++)
	{
		neg[i] = -1*vector[i];
	}
	add(neg, m, axis);
	delete[] neg;
}



/**
compute the weighted average value of each row (or column, specified by axis)
@param axis row by row (0) or column by column (1)
@param  weights optional array of weights for weighted average (if NULL, will be taken as all 1's).  (should be length numCols() for row-by-row, and length numRows() for col-by-col)
*/
double* Matrix::average(int axis, double *weights)
{
	// determine appropriate length for result and weights
	int wtlen=0,reslen=0;
	if (axis==0) 
	{
		wtlen=numCols; 
		reslen=numRows;
	}
	else 
	{
		wtlen=numRows;
		reslen = numCols;
	}
	double* result = new double[reslen];
	bool allocated=false;
	// if weights are NULL, fill in equal weights
	if (weights==NULL)
	{
		weights = new double[wtlen];
		allocated = true;
		for (int i=0; i<wtlen; i++) weights[i] =1;
	}
	// normalize weights so they sum to 1, in case caller didn't
	double wtSum=0;
	for (int i=0; i<wtlen; i++) wtSum+=weights[i];
	for (int i=0; i<wtlen; i++) weights[i] = weights[i]/wtSum;

	// now do the weighted averaging
	if (axis==0) //row by row
	{
		for (int i=0; i<numRows; i++) // row i
		{
			double sum=0;
			for (int j=0; j<numCols; j++) // column j
			{
				//printf("sum %f, adding %f *%f\n", sum, weights[j], columns[j][i]);
				sum += weights[j]*columns[j][i];
			}
			result[i] = sum;
		}
	}
	else if (axis==1) // column by column
	{
		for (int j=0; j<numCols; j++) // column j
		{
			double sum=0;
			for (int i=0; i<numRows; i++) // row i
			{
				sum+= weights[i]*columns[j][i];
			}
			result[j] = sum;
		}
	}
	if (allocated) delete[] weights; // delete the weights only if they were created locally
	return result;
}

// update the column-major array representation of matrix, in case the matrix has changed since it was last computed
void Matrix::updateArray() 
{
	if (changed)
	{	
		delete[] entries; // this may be causing trouble?
		entries = new double[numRows*numCols];
		for (int j=0; j< numCols; j++)
		{
			for (int i=0; i<numRows; i++)
			{
				entries[j*numRows+i] = columns[j][i];
			}
		}
		changed=false;
	}
}

/**
Print out the matrix
*/
void Matrix::print()
{
	printf("\n-----------MATRIX---------------------\n");
	printf("%d rows, %d columns\n", numRows, numCols);
	printf("%d columns \n", columns.size());
	fflush(stdout);
	for (int i=0; i<numRows; i++)
	{
		for (int j=0; j<numCols; j++)
		{
			printf(" %f", columns[j][i]);
		}
		printf("\n");
		fflush(stdout);
	}
	printf("--------------------------------------\n");
}

Matrix::~Matrix()
{
	delete[] entries;
}

void Matrix::clear()
{

	this->columns.clear();

	delete[] entries;
	entries = new double[0];
	changed = true;
	numRows=0;
	numCols=0;
}
// find row i and column j corresponding to array index k in column major array format
void matrixIndex(int k, int dim, int &i, int &j)
{
	j = k/dim;
	i = k%dim;
}

// this is code to put a 2-dimensional matrix in column-major form
/*for (int i=0; i<N*N; i++)
	{
		for (int j=0; j<N; j++)
		{
			A[index++] = matrix[j][i]; // column major is more efficient with LAPACK
		}
	}*/



// g++ Matrix.cc -I/home/rborbely/lapack-3.4.1/lapacke/include /usr/lib64/liblapack.so /home/rborbely/lapack-3.4.1/liblapacke.a /usr/lib64/libblas.a

