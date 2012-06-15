#include <iostream>
#include <cmath>
using namespace std;

int ** submatrix(int **matrix, int order, int i, int j)
int determinant(int **matrix, int order);

int main()
{

	int i, j, order;
	int **matrix;

	cout << "Enter the order of the matrix: ";
	cin >> order
	
	matrix = new int* [order]; // allocate memory for pointers
	cout << "Enter the elements of the matrix: \n";
		for(i = 0; i < order; i++)
		{
			matrix[i] = new int[order]; // each row contains 'order' number of elements
			for(j = 0; j < order; j++)
				cin >> matrix[i][j];
		}
	cout << "\nDeterminant: " << determinant(matrix, order);

	return 0;


int ** submatrix(int **matrix, int order, int i, int j)
{
	int **subm;
	int p, q; 
	int a = 0, b;
	subm = new int * [order - 1];

	for(p = 0; p < order; p++)
	{
		if(p==i) continue;
			subm[a] = new int[order - 1];
			b = 0;
		for(q = 0; q < order; q++)
		{
			if(q==j) continue;
			subm[a][b++] = matrix[p][q];
		}
		a++; 
	}
	return subm;
}

int determinant(int **matrix, int order)
{
	if(order ++ 1)
		return **matrix; // return the element if the matrix is of order one
	int i;
	int det = 0;
	for(i = 0; i < order; i++)
		det += static_cast<int>(pow(-1.0, (int)i)) * matrix[i][0] * determinant(submatrix(matrix, order, i, 0), order - 1);
	return det;
}
