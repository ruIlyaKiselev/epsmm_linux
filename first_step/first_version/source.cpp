#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <stdio.h>

inline double calculateP(int j, int Nx) {
    return (j < Nx / 2.0) ? pow(0.1, 2) : pow(0.2, 2);
}

inline double calculateTao(int Nx, int Ny) {
    return (Nx <= 1000 && Ny <= 1000) ? 0.01 : 0.001;
}

inline void initArrays(int Nx, int Ny, double* U0, double* U1, double* U2, double* P) {
    for (int i = 0; i != Nx; i++) {
        for (int j = 0; j != Ny; j++) {
            *(U0 + i * Ny + j) = 0;
            *(U1 + i * Ny + j) = 0;
            *(U2 + i * Ny + j) = 0;
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
        return exp(-pow(2 * M_PI * f0 * (n * tao - t0), 2) / pow(gamma, 2)) * sin(2 * M_PI * f0 * (n * tao - t0));;
    }

    return 0;
}

void algorithmStep(int i, int j, int n, int Nx, int Ny, int Sx, int Sy, double Hx, double Hy, double* U0, double* U1, double* U2, double* P) {
    double tao = calculateTao(Nx, Ny);

    double leftPartX = (*(U1 + i * Ny + j + 1) - *(U1 + i * Ny + j)) * (*(P + (i - 1) * Ny + j) + *(P + i * Ny + j));
    double rightPartX = (*(U1 + i * Ny + j - 1) - *(U1 + i * Ny + j)) * (*(P + (i - 1) * Ny + j - 1) + *(P + i * Ny + j - 1));
    double leftPartY = (*(U1 + (i + 1) * Ny + j) - *(U1 + i * Ny + j)) * (*(P + i * Ny + j - 1) + *(P + i * Ny + j));
    double rightPartY = (*(U1 + (i - 1) * Ny + j) - *(U1 + i * Ny + j)) * (*(P + (i - 1) * Ny + j - 1) + *(P + (i - 1) * Ny + j));

    double partX = (leftPartX + rightPartX) / (2 * pow(Hx, 2));
    double partY = (leftPartY + rightPartY) / (2 * pow(Hy, 2));

    *(U2 + i * Ny + j) = 2 * *(U1 + i * Ny + j) - *(U0 + i * Ny + j) + pow(tao, 2) * (functionF(i, j, n, Nx, Ny, Sx, Sy) + partX + partY);
}

void processAlgorithm(int Nx, int Ny, int Nt, int Sx, int Sy, double Hx, double Hy, double* U0, double* U1, double* U2, double* P) {
    for (int n = 1; n != Nt; n++) {
        //double Umax = 0.0;
        for (int i = 1; i != Nx - 2; i++) {
            for (int j = 1; j != Ny - 2; j++) {
                if (n % 3 == 1) {
                    algorithmStep(i, j, n, Nx, Ny, Sx, Sy, Hx, Hy, U0, U1, U2, P);
                    //if ((abs(*(U2 + (i - 1) * Ny + j)) > Umax)) {
                    //    Umax = abs((*(U2 + (i - 1) * Ny + j)));
                    //}
                    continue;
                } else if (n % 3 == 2) {
                    algorithmStep(i, j, n, Nx, Ny, Sx, Sy, Hx, Hy, U1, U2, U0, P);
                    //if ((abs(*(U0 + (i - 1) * Ny + j)) > Umax)) {
                    //    Umax = abs((*(U0 + (i - 1) * Ny + j)));
                    //}
                    continue;
                } else if (n % 3 == 0) {
                    algorithmStep(i, j, n, Nx, Ny, Sx, Sy, Hx, Hy, U2, U0, U1, P);
                    //if ((abs(*(U1 + (i - 1) * Ny + j)) > Umax)) {
                    //    Umax = abs((*(U1 + (i - 1) * Ny + j)));
                    //}
                }
            }
        }
        //printf("%.36lf\n", Umax);
    }
}

void saveArrayToFile(int Nx, int Ny, const double* dataArray, const char* filename) {
    FILE* file = fopen(filename, "wb");
    fwrite(dataArray, sizeof(double), Nx * Ny, file);
    fclose(file);
}

int main(int argc, char* argv[]) {
    int Nx = 10000, Ny = 10000; // mash size
    int Nt = 100; // step count

    double Xa = 0.0, Xb = 4.0;
    double Ya = 0.0, Yb = 4.0;

    double Hx = (Xb - Xa) / (Nx - 1);
    double Hy = (Yb - Ya) / (Ny - 1);

    int Sx = 1;
    int Sy = 1;

    double* U0 = (double*)malloc(Nx * Ny * sizeof(double));
    double* U1 = (double*)malloc(Nx * Ny * sizeof(double));
    double* U2 = (double*)malloc(Nx * Ny * sizeof(double));
    double* P = (double*)malloc(Nx * Ny * sizeof(double));

    initArrays(Nx, Ny, U0, U1, U2, P);
    processAlgorithm(Nx, Ny, Nt, Sx, Sy, Hx, Hy, U0, U1, U2, P);
    saveArrayToFile(Nx, Ny, (Nt % 3 == 1) ? U2 : (Nt % 3 == 2) ? U0 : U1, "double00001500.dat");

    free(U0);
    free(U1);
    free(U2);
    free(P);
}
