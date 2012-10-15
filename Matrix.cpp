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


/*! \file Matrix.cpp
*   \brief Matrix class method implementations.
*/
#include <lapacke.h>
#include <stdio.h>
#include <vector>
#include <omp.h>

#include <iostream>



#include "Matrix.h"

#define matrix_debug 0


using namespace std;

/** \brief cmIndex compute array offset
 * @param i 0-rel column index
 * @param j 0-rel row index
 * @param dim dimensionality of data
 * @return offset within 1-d array representation in col-major format
 */
int cmIndex(int i, int j, int dim)
{
	return j*dim + i;
}


/** \brief Matrix create an empty matrix
*/
Matrix::Matrix()
{ 

	entries = new double[0];
	changed = false;
	numRows=0;
	numCols=0;
}

/** \brief Matrix create matrix of zeroes, n rows, m columns
* @param n number of rows
* @param m number of columns
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

		std::vector<double> temp;
		columns.push_back(temp);

		for (int i=0; i<n; i++)
		{
			columns[j].push_back(0);
		}
	}
	
}

/** \brief Matrix Constructor to initialize matrix to 0's off-diagonal, and  a given vector on the diagonal (a la numpy.diag)
* @param di vector of length len, to be used as the diagonal
* @param len length of di
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

/** \brief Matrix(double array, num rows, num cols, orientation)
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

/** \brief get value of matrix element
@param i the 0-rel row number
@param j the 0-rel column number
@return the element in ith row, jth column (indexed from 0)
*/
double Matrix::getValue(int i, int j) const
{
	std::vector<double> col = columns[j];
	return col[i];
}

/**
\brief How many rows are in the matrix?
@return the number of rows
*/
int Matrix::rowCount() const
{
	return numRows;
}

/**
\brief How many columns are in the matrix?
@return the number of columns
*/
int Matrix::colCount() const
{
	return numCols;
}

/**
 * \brief assign value to matrix cell
@param val the value
@param i 0-rel row number
@param j 0-rel column number
*/
void Matrix::assign(double val, int i, int j)
{
	if (j>=numCols || j<0) throw SizeError((char *)"Error: attempt to assign value to a non-existent column");
	if (i>=numRows || i<0) throw SizeError((char *)"Error: attempt to assign value to a non-existent row");
	columns[j].insert(columns[j].begin() + i, val);
	changed = true;
}

/** \brief Overwrite val in the ith row, jth column of the matrix
@param val double value to write
@param i 0-rel row number
@param j 0-rel column number
*/
void Matrix::update(double val, int i, int j)
{
	if (j>=numCols || j<0) throw SizeError((char *)"Error: attempt to write value to a non-existent column");
	if (i>=numRows || i<0) throw SizeError((char *)"Error: attempt to write value to a non-existent row");
	columns[j][i] = val;
	changed = true;
}




/**
\brief Insert row as the rowNum'th row in the matrix (indexed from 0)
@param row the row to insert 
@param rowSize the length of row (should be the same as colCount())
@param rowNum location at which to insert row -- row will be the rowNum'th row in the matrix (indexed from 0) (0 <= rowNum <= rowCount())
*/
Matrix & Matrix::insertRow(double * row, int rowSize, int rowNum) throw (SizeError)
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
\brief Insert col as the colNum'th column in the matrix (indexed from 0)
@param col the column to insert
@param colSize the length of col (should be the same as numRows())
@param colNum location at which to insert col -- col will be the colNum'th column in the matrix (indexed from 0)
(0  <= colNum <= colCount())
*/
Matrix & Matrix::insertColumn(double * col, int colSize, int colNum) throw (SizeError)
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


/** \brief return a copy of rowOffset'th row of the matrix
@param rowOffset number of the row to retrieve (indexed from 0)
@param vec empty  vector in whuch to return row data
@return ref to vector representation of the specified row*/
std::vector<double> & Matrix::getCopyOfRow(int rowOffset,std::vector<double> & vec) throw (SizeError)
{
	if (rowOffset<0 || rowOffset > numRows)
		throw SizeError((char*)"Error: attempted to get copy of non-existent row");

	for (int j=0; j<numCols; j++)
		vec.push_back(columns[j][rowOffset]);
	return vec;
}

/** \brief return a copy of colOffset'th row of the matrix
@param colOffset number of the column to retrieve (indexed from 0)
@param vec empty  vector in whuch to return row data
@return ref to vector representation of the specified column*/
std::vector<double> & Matrix::getCopyOfColumn(int colOffset, std::vector<double> & vec) throw (SizeError)
{
	if (colOffset<0 || colOffset > numCols)
		throw SizeError((char*)"Error: attempted to get copy of non-existent column");

	for (int i=0; i<numRows; i++)
	{
		vec.push_back(columns[colOffset][i]);
	}
	return vec;
}


// this is only used as a dummy function for LAPACK
lapack_logical leq(const double* a, const double* b)
{
	if (*a<=*b) return 1;
	else return 0;
}


/**
* \brief det
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
\brief Matrix multiplication this*B
@param B matrix to multiply by
@return product of matrix multiplication: this*B. caller must delete.
*/
Matrix & Matrix::dot(Matrix& B) const
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

/** \brief Subtract Matrix B from this Matrix
	@param B Matrix to subtract from this. caller must delete.
*/
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
\brief matrix multiplication for matrices already represented in column-major form
@param A col-major matrix
@param B col-major matrix
@param rows1 rows for A, cols for B
@param cols1 cols for A
@param cols2 cols for B
@param[out] result column-major result array (caller allocates)
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
\brief Compute the covariance of a positive, definite matrix (numpy.covar)
@return the covariance matrix. caller deletes.
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
\brief Invert the matrix
@return the inverse of this matrix. caller deletes.
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
\brief Add vector to matrix row by row or column by column (in place)
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
\brief Subtract vector from matrixArray row by row or column by column
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
\brief compute the weighted average value of each row (or column, specified by axis)
@param axis row by row (0) or column by column (1)
@param  weights optional array of weights for weighted average (if NULL, will be taken as all 1's).  (should be length numCols() for row-by-row, and length numRows() for col-by-col)
@return prt to averages (caller deletes)
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

/**
\brief update the column-major array representation of matrix, in case the matrix has changed since it was last computed
*/
void Matrix::updateArray() 
{
	if (changed)
	{	
		delete[] entries;
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
\brief Print out the matrix
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
			printf(" %lf", columns[j][i]);
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

/**
 * \brief free the matrix resouces
 */
void Matrix::clear()
{

	this->columns.clear();

	delete[] entries;
	entries = new double[0];
	changed = true;
	numRows=0;
	numCols=0;
}
/**
 \brief find row i and column j corresponding to array index k in column major array format
*/
void matrixIndex(int k, int dim, int &i, int &j)
{
	j = k/dim;
	i = k%dim;
}

