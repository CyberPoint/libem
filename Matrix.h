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


/*! \file Matrix.h
*   \brief Definitions for a Matrix class wrapping lapack/blas matrix routimes
*/
#ifndef MATRIX_HEADER_H
#define MATRIX_HEADER_H

#include <vector>
#include <string>
#include <exception>
#include <stdexcept>


//! class for returning errors due to mismatched size in matrix operations */
class SizeError: public std::runtime_error
{
	public:
	SizeError(const char* error="Matrix operation could not be performed due to a size mismatch."):std::runtime_error(error){}
};


//! Exception thrown when there is an error in a call through lapacke */
class LapackError: public std::runtime_error
{
	public:
	LapackError(const char* error="Operation could not be performed--error in call to lapacke"):std::runtime_error(error){}
};


class Matrix
{
	public:
	/** enum for specifying how a matrix is represented as an array - used for matrix constructor*/
	enum Orientation{ROW_MAJOR, COLUMN_MAJOR };

	/**create an empty matrix (size 0x0) */
	Matrix();

	/** create a matrix of zeroes, of size nxm 
	@param n number of rows
	@param m number of columns*/
	Matrix(int n, int m);

	/**Constructor to initialize matrix to 0's off-diagonal, and a given vector on the diagonal (a la numpy.diag)
	@param di -- vector of length len, to be used as the diagonal
	param len length of di*/
	Matrix(double di[], int len);

	/**Create a matrix from its column-major or row-major matrix representation
	@param a array of row-major or column-major representation of matrix
	@param nRows current number of rows
	@param nCols current number of columns
	@param major Orientation.ROW_MAJOR or Orientation.COLUMN_MAJOR*/
	Matrix(double a[], int numRows, int numCols, Orientation major);

	/** Create a matrix from its serialization
	    @param array A serialization created by Matrix::serialize()
	*/
	Matrix(double *array);

	/** Create a serialization of the matrix
	    @param a array of row-major or column-major representation of matrix
	    @return a serialization of the array
	*/
	double * Serialize();

        /** \brief Matrix(double array)
	    Fill a matrix from a Matrix serialization
	    @param array A serialization created by Matrix::serialize()
	*/
	void deSerialize(double *array);

	/**Assign val to the ith row, jth column of the matrix
	@param val double value to insert
	@param i row number
	@param j column number*/
	void assign(double val, int i, int j);

	/**Overwrite val in the ith row, jth column of the matrix
	@param val double value to write
	@param i 0-rel row number
	@param j 0-rel column number*/
	void update(double val, int i, int j);



	/**Insert row as the rowNum'th row in the matrix (indexed from 0)
	@param row the row to insert 
	@param rowSize the length of row (should be the same as colCount() for non-empty matrix)
	@param rowNum location at which to insert row -- row will be the rowNum'th row 
		in the matrix (indexed from 0) (0 <= rowNum <= rowCount())*/
	Matrix &  insertRow(double * row, int rowSize, int rowNum) throw (SizeError);

	/**Insert col as the colNum'th column in the matrix (indexed from 0)
	@param col the column to insert
	@param colSize the length of col (should be the same as numRows() for non-empty matrix)
	@param colNum location at which to insert col -- col will be the colNum'th column 
		in the matrix (indexed from 0) (0  <= colNum <= colCount())*/
	Matrix &  insertColumn(double * col, int colSize, int colNum) throw (SizeError);
	
	/** How many rows are in the matrix?
	@return the number of rows*/
	int rowCount() const;
	
	/**  How many columns are in the matrix?
	@return the number of columns*/
	int colCount() const;

	/**@return the element in ith row, jth column (indexed from 0)*/
	double getValue(int row, int column) const;

	/**Invert the matrix
	@return the inverse of this matrix*/
	Matrix * inv() throw (SizeError, LapackError);

	/**Matrix multiplication -- this*B
	@param B matrix to multiply by
	@return product of matrix multiplication: this*B  */
	Matrix * dot(const Matrix& m) const;

	/**@return the determinant. Note: only works for square matrices*/
	double det() throw (LapackError, SizeError);

	/**return a copy of rowOffset'th row of the matrix
	@param rowOffset number of the row to retrieve (indexed from 0)
	@param vec empty  vector in whuch to return row data
	@return ref to vector representation of the specified row*/
	std::vector<double> & getCopyOfRow(int rowOffset,std::vector<double> & vec) throw (SizeError);

	/** return a copy of colOffset'th row of the matrix
	@param colOffset number of the column to retrieve (indexed from 0)
	@param vec empty  vector in whuch to return row data
	@return ref to vector representation of the specified column*/
	std::vector<double> & getCopyOfColumn(int colOffset, std::vector<double> & vec) throw (SizeError);

	/**  Add vector to matrix row by row or column by column (in place)
	@param vector array to add
	@param m length of vector
	@param axis row by row (0) or column by column (1)*/
	void add(double vector[], int m, int axis);

	/** Subtract Matrix B from this Matrix 
	@param B Matrix to subtract from this*/
	Matrix* subtract(const Matrix& B) const;

	/**  Subtract vector from matrix row by row or column by column (in place)
	@param vector array to subtract
	@param m length of vector
	@param axis row by row (0) or column by column (1)*/
	void subtract(double vector[], int m, int axis);

	/** Print the matrix to stdout */
	void print();
	
	/** clear all entries in the matrix. leaves an empty 0 x 0 matrix. */
	void clear();

	/** Destuctor */
	~Matrix();

	private:
	std::vector< std::vector<double> > columns; ///columns of matrix--this is kept up to date as matrix is modified
	
	/// representation of matrix as 1-d array in column major order.  This is not kept up to date as
	/// matrix is created/modified, but only updated when needed for an operation
	double * entries ; 

	void updateArray(); ///< update column-major representation in entries

	bool changed;//< has matrix been modified since last call to updateArray()?
	int numRows; ///< number of rows in matrix
	int numCols; ///< number of columns in matrix

	/// matrix mult A*B, where A and B are matrices represented in column-major form, with dimensions
	/// rows1xcols1 and cols1xcols2 respectively.
	void dot(double A[], double B[], int rows1, int cols1, int cols2, double result[]);
};
#endif //MATRIX_HEADER
