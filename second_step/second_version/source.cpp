#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <xmmintrin.h>

inline double calculateP(int j, int Nx) {
    return (j < Nx / 2.0) ? 0.01 : 0.04;
}

inline double calculateTao(int Nx, int Ny) {
    return (Nx <= 1000 && Ny <= 1000) ? 0.01 : 0.001;
}

inline void initArrays(const int Nx, const int Ny, double* U0, double* U1, double* P) {
    for (int i = 0; i != Nx; i++) {
        for (int j = 0; j != Ny; j++) {
            *(U0 + i * Ny + j) = 0;
            *(U1 + i * Ny + j) = 0;
            *(P + i * Ny + j) = calculateP(j, Nx);
        }
    }
}

double functionF(int i, int j, int n, int Nx, int Ny, int Sx, int Sy) {
    double f0 = 1.0;
    double t0 = 1.5;
    double gamma = 4.0;
    double tao = calculateTao(Nx, Ny);

    if (j == Sx && i == Sy) {
        return exp(-((2 * M_PI * f0 * (n * tao - t0)) * (2 * M_PI * f0 * (n * tao - t0))) / (gamma * gamma)) * sin(2 * M_PI * f0 * (n * tao - t0));
    }

    return 0;
}

void algorithmStep(int i, int j, int n, int Nx, int Ny, int Sx, int Sy, double Hx, double Hy, double* U0, double* U1, double* U2, double* P) {
    double tao = calculateTao(Nx, Ny);
    
    __m128d *vu0 = (__m128d*)U0;
    __m128d *vu1 = (__m128d*)U1;
    __m128d *vu2 = (__m128d*)U2;
    
    double U1ij = *(U1 + i * Ny + j); // we use it 5 times (5 memory access operation), in this case we use memory access 1 times (5 times instead)
    double Pij = *(P + i * Ny + j); // it works for other matrix elements from P, we take value by address once, but use it 2 times
    double Pim1jm1 = *(P + (i - 1) * Ny + j - 1); 
    double Pim1j = *(P + (i - 1) * Ny + j);
    double Pijm1 = *(P + i * Ny + j - 1);

    double leftPartX = (*(U1 + i * Ny + j + 1) - U1ij) * (Pim1j + Pij);
    double rightPartX = (*(U1 + i * Ny + j - 1) - U1ij) * (Pim1jm1 + Pijm1);
    double leftPartY = (*(U1 + (i + 1) * Ny + j) - U1ij) * (Pijm1 + Pij);
    double rightPartY = (*(U1 + (i - 1) * Ny + j) - U1ij) * (Pim1jm1 + Pim1j);

    double partX = (leftPartX + rightPartX) * Hx;
    double partY = (leftPartY + rightPartY) * Hy;

    *(U2 + i * Ny + j) = 2 * *(U1 + i * Ny + j) - *(U0 + i * Ny + j) + tao * tao * (functionF(i, j, n, Nx, Ny, Sx, Sy) + partX + partY);
}

void processAlgorithm(int Nx, int Ny, int Nt, int Sx, int Sy, double Hx, double Hy, double* U0, double* U1, double* U2, double* P) {
    for (int n = 1; n != Nt; n++) {
        //double Umax = 0.0;
        for (int i = 1; i != Nx - 2; i++) {
            for (int j = 1; j != Ny - 2; j++) {
                algorithmStep(i, j, n, Nx, Ny, Sx, Sy, Hx, Hy, U0, U1, U2, P);
                //Umax = (fabs(*(U2 + i * Ny + j)) > Umax) ? fabs(*(U2 + i * Ny + j)) : Umax;
            }
        }
        double* tmp = U0;
        U0 = U1;
        U1 = U2;
        U2 = tmp;
        //printf("%.16lf\n", Umax);
    }
}

void saveArrayToFile(int Nx, int Ny, const double* dataArray, const char* filename) {
    FILE* file = fopen(filename, "wb");
    fwrite(dataArray, sizeof(double), Nx * Ny, file);
    fclose(file);
}

int main(int argc, char* argv[]) {
    const int Nx = 10000, Ny = 10000;
    const int Nt = 100;

    const double Xa = 0.0, Xb = 4.0;
    const double Ya = 0.0, Yb = 4.0;

    const double Hx = 1 / (2 * pow((Xb - Xa) / (Nx - 1), 2));
    const double Hy = 1 / (2 * pow((Yb - Ya) / (Ny - 1), 2));

    const int Sx = 1;
    const int Sy = 1;

    double* U0 = (double*)malloc(Nx * Ny * sizeof(double));
    double* U1 = (double*)malloc(Nx * Ny * sizeof(double));
    double* U2 = (double*)malloc(Nx * Ny * sizeof(double));
    double* P = (double*)malloc(Nx * Ny * sizeof(double));

    initArrays(Nx, Ny, U0, U1, P);
    processAlgorithm(Nx, Ny, Nt, Sx, Sy, Hx, Hy, U0, U1, U2, P);
    saveArrayToFile(Nx, Ny, U2, "double00001500.dat");

    free(U0);
    free(U1);
    free(U2);
    free(P);
}
