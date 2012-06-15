#include <iostream>
#include <vector>
using namespace std;

// using integer arrays
void transpose(double *psrc, int& R, int& C)
{
	double *pcopy = new double[R*C];
	for(int j = 0; j < R*C; j++)
		pcopy[j] = psrc[j]; // copy source
	// overwrite source from copy
	int r = 0, c = 0, tidx = 0;
	for (int j = 0; j < R*C; j++)
	{
		r = j/C;
		c = j%C;
		tidx = c*R + r;
		psrc[tidx] = pcopy[j];
	}
	delete [] pcopy;

	return;
}

void arrayout(double *pi, int& R, int& C)
{
	for(int r = 0; r < R; r++)
	{
		for(int c = 0; c < C; c++)
			cout << pi[r*C+c] << " "; 		
			cout << endl;
	}
	return;
}

// using vectors
void transposev(vector<double>& psrc, int& R, int& C)
{
	vector<double> pcopy;
	pcopy.resize(R*C);

	for(int j = 0; j < R*C; j++)
		pcopy[j] = psrc[j]; // copy source
	// overwrite source from copy
	int r = 0, c = 0, tidx = 0;
	for(int j = 0; j < R*C; j++)
	{
		r = j/C;
		c = j%C;
		tidx = c*R + r;
		psrc[tidx] = pcopy[j];
	}
	return;
}
void vectorout(vector<double>& pi, int& R, int& C)
{
	for(int r = 0; r < R; r++)
	{
		for(int c = 0; c < C; c++)
			cout << pi[r*C+c] << " ";
		cout << endl;
	}
	return;
}

int main()
{
	int R = 1, C = 1;

	cout << "Enter rows: ";
	cin >> R;
	cout << "Enter cols: ";
	cin >> C;

	cout << "using vectors" << endl << endl;
	vector<double> vdbl;
	vdbl.resize(R*C);
	for(int j = 0; j < R*C; j++_
		vdbl[j] = 10+(double)j; // elements assigned sequential values for now
	vectorout(vdbl, R, C); // show as R rows, C columns
	transposev(vdbl, R, C); // transpose elements
	cout << endl;
	vectorout(vdbl, C, R); // show as C rows, R columns
	cout << endl;

	cout << "using dynamic arrays" << endl << endl;
	double* p_dbl = new double[R*C];
	for(int j = 0; j < R*C; j++)
		p_dbl[j] = 10+(double)j; // elements assigned here
	arrayout(p_dbl, R, C);
	transpose(p_dbl, R, C);
	cout << endl;
	arrayout(p_dbl, C, R);
	delete [] p_dbl;

	system("PAUSE");
	return 0;
}
















