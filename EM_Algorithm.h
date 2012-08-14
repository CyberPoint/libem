#ifndef EM_AGORITHM_H
#define EM_ALGORITHM_H

/* EXPECTATION MAXIMIZATION ALGORITHM (header file)
   CYBERPOINT INTERNATIONAL, LLC
   Written by Elizabeth Garbee, Summer 2012 */

#include "/home/egarbee/gmm/Matrix.h"

using namespace std;

/************************************************************************************************
** EM FUNCTION DECLARATIONS **
************************************************************************************************/

double estep(int n, int m, int k, double *X,  Matrix &p_nk_matrix, vector<Matrix *> &sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix);

bool mstep(int n, int m, int k, double *X, Matrix &p_nk_matrix, Matrix *sigma_matrix, Matrix &mu_matrix, Matrix &Pk_matrix);

void EM(int n, int m, int k, double *X, vector<Matrix*> &sigma_matrix, Matrix &mu_matrix, Matrix &Pks);

#endif //EM_ALGORITHM_HEADER

