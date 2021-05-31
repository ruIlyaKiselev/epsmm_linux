#include <stdio.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

void swapArrays(double *U0, double *U1) {
    double *tmp = U0;
    U0 = U1;
    U1 = tmp;
}

double calculateP(int j, int Nx) {
    return (j < Nx / 2.0) ? 0.01 : 0.04;
}

double calculateTao(int Nx, int Ny) {
    return (Nx <= 1000 && Ny <= 1000) ? 0.01 : 0.001;
}

void initArrays(const int Nx, const int Ny, double* U0, double* U1, double* P) {
    for (int i = 0; i != Nx; i++) {
        for (int j = 0; j != Ny; j++) {
            *(U0 + i * Ny + j) = 0;
            *(U1 + i * Ny + j) = 0;
            *(P + i * Ny + j) = calculateP(j, Nx);
        }
    }
}

double functionF(int n, int Nx, int Ny) {
    double f0 = 1.0;
    double t0 = 1.5;
    double gamma = 4.0;
    double tao = calculateTao(Nx, Ny);

    return exp(-((2 * M_PI * f0 * (n * tao - t0)) * (2 * M_PI * f0 * (n * tao - t0))) / (gamma * gamma)) *
           sin(2 * M_PI * f0 * (n * tao - t0));
}

void saveArrayToFile(int Nx, int Ny, const double* dataArray, const char* filename) {
    FILE* file = fopen(filename, "wb");
    fwrite(dataArray, sizeof(double), Nx * Ny, file);
    fclose(file);
}

void algorithmStepVector(int Nx, int Ny, double tao2Hx2, double tao2Hy2, double* pU0, double* pU1, double* pP) {

    __m256d U1ijV = _mm256_loadu_pd(pU1);
    __m256d PijV = _mm256_loadu_pd(pP);
    __m256d Pim1jm1V = _mm256_loadu_pd(pP - Nx - 1);
    __m256d Pim1jV = _mm256_loadu_pd(pP - Nx);
    __m256d Pijm1V = _mm256_loadu_pd(pP - 1);

    __m256d leftPartX = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(pU1 + 1), U1ijV), _mm256_add_pd(Pim1jV, PijV));
    __m256d rightPartX = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(pU1 - 1), U1ijV), _mm256_add_pd(Pim1jm1V, Pijm1V));
    __m256d leftPartY = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(pU1 + Nx), U1ijV), _mm256_add_pd(Pijm1V, PijV));
    __m256d rightPartY = _mm256_mul_pd(_mm256_sub_pd(_mm256_loadu_pd(pU1 - Nx), U1ijV),_mm256_add_pd(Pim1jm1V,Pim1jV));

    __m256d partsSum = _mm256_fmadd_pd( _mm256_add_pd(leftPartX, rightPartX), _mm256_set1_pd(tao2Hx2),
                                        _mm256_mul_pd(_mm256_add_pd(leftPartY, rightPartY), _mm256_set1_pd(tao2Hy2)));

    _mm256_storeu_pd(pU0, _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(U1ijV, U1ijV), _mm256_load_pd(pU0)), partsSum));
    //__m256d partX = _mm256_mul_pd(_mm256_add_pd(leftPartX, rightPartX), _mm256_set1_pd(tao2Hx2));
    //__m256d partY = _mm256_mul_pd(_mm256_add_pd(leftPartY, rightPartY), _mm256_set1_pd(tao2Hy2));

    //_mm256_storeu_pd(pU0, _mm256_add_pd(_mm256_sub_pd(_mm256_add_pd(U1ijV, U1ijV), _mm256_load_pd(pU0)), _mm256_add_pd(partX, partY)));
}

void algorithmStepScalar(int Nx, int Ny, double tao2Hx2, double tao2Hy2, double* pU0, double* pU1, double* pP) {

    double U1ij = *pU1;
    double Pij = *pP;
    double Pim1jm1 = *(pP - Nx - 1);
    double Pim1j = *(pP - Nx);
    double Pijm1 = *(pP - 1);

    double leftPartX = (*(pU1 + 1) - U1ij) * (Pim1j + Pij);
    double rightPartX = (*(pU1 - 1) - U1ij) * (Pim1jm1 + Pijm1);
    double leftPartY = (*(pU1 + Nx) - U1ij) * (Pijm1 + Pij);
    double rightPartY = (*(pU1 - Nx) - U1ij) * (Pim1jm1 + Pim1j);

    double partX = (leftPartX + rightPartX) * tao2Hx2;
    double partY = (leftPartY + rightPartY) * tao2Hy2;

    *pU0 = 2 * U1ij - *pU0 + partX + partY;
}

void finalStep(int Nx, int Ny, int Sx, int Sy, double tao2functionF, double tao2Hx2, double tao2Hy2, double* U0, double* U1, double* P) {

    double U1ij = *(U1 + Sy * Nx + Sx);
    double Pij = *(P + Sy * Nx + Sx);
    double Pim1jm1 = *(P + (Sy - 1) * Nx + Sx - 1);
    double Pim1j = *(P + (Sy - 1) * Nx + Sx);
    double Pijm1 = *(P + Sy * Nx + Sx - 1);

    double leftPartX = (*(U1 + Sy * Nx + Sx + 1) - U1ij) * (Pim1j + Pij);
    double rightPartX = (*(U1 + Sy * Nx + Sx - 1) - U1ij) * (Pim1jm1 + Pijm1);
    double leftPartY = (*(U1 + (Sy + 1) * Nx + Sx) - U1ij) * (Pijm1 + Pij);
    double rightPartY = (*(U1 + (Sy - 1) * Nx + Sx) - U1ij) * (Pim1jm1 + Pim1j);

    double partX = (leftPartX + rightPartX) * tao2Hx2;
    double partY = (leftPartY + rightPartY) * tao2Hy2;

    *(U0 + Sy * Nx + Sx) = 2 * U1ij - *(U0 + Sy * Nx + Sx) + partX + partY + tao2functionF;
}

void processAlgorithm(int Nx, int Ny, int Nt, int Sx, int Sy, double tao2Hx2, double tao2Hy2, double tao, double* U0, double* U1, double* P) {

#pragma omp parallel proc_bind(close)
    for (int n = 0; n != Nt; n++) {
        double tao2functionF = tao * tao * functionF(n, Nx, Ny);

        double *pU0 = U0 + Nx + 1;
        double *pU1 = U1 + Nx + 1;
        double *pP = P + Nx + 1;

#pragma omp for schedule(static, Ny / omp_get_num_threads())
        for (int i = 1; i != Ny - 1; i++) {
            int j;
            for (j = 1; j <= Nx - 4 - 1; j += 4) {
                algorithmStepVector(Nx, Ny, tao2Hx2, tao2Hy2, pU0, pU1, pP);

                pU0 += 4;
                pU1 += 4;
                pP += 4;
            }

            while (j != Nx - 1) {
                algorithmStepScalar(Nx, Ny, tao2Hx2, tao2Hy2, pU0, pU1, pP);
                pU0++;
                pU1++;
                pP++;
                j++;
            }

            pU0 += 2;
            pU1 += 2;
            pP += 2;
        }

        finalStep(Nx, Ny, Sx, Sy, tao2functionF, tao2Hx2, tao2Hy2, U0, U1, P);
#pragma omp single
        {
            double *tmp = U0;
            U0 = U1;
            U1 = tmp;
        };
#pragma omp barrier
    }
}

int main(int argc, char **argv) {
    int Nx = 10000, Ny = 10000;
    int Nt = 100;

    if (argc == 5) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nt = atoi(argv[3]);
        omp_set_num_threads(atoi(argv[4]));
    }

    const double tao = calculateTao(Nx, Ny);

    const double Xa = 0.0, Xb = 4.0;
    const double Ya = 0.0, Yb = 4.0;

    const double Hx2 = (2 * pow((Xb - Xa) / (Nx - 1), 2));
    const double Hy2 = (2 * pow((Yb - Ya) / (Ny - 1), 2));

    const double tao2Hx2 = tao * tao / Hx2;
    const double tao2Hy2 = tao * tao / Hy2;

    const int Sx = 1;
    const int Sy = 1;

    double* U0;
    double* U1;
    double* P;

    int result;

    result = posix_memalign((void**)&U0, sizeof(double) * 4, Nx * Ny * sizeof(double));
    if (result != 0) {
        printf("Problem in posix_memalign for U0 array. Error code: %d", result);
    }

    result = posix_memalign((void**)&U1, sizeof(double) * 4, Nx * Ny * sizeof(double));
    if (result != 0) {
        printf("Problem in posix_memalign for U1 array. Error code: %d", result);
    }

    result = posix_memalign((void**)&P, sizeof(double) * 4, Nx * Ny * sizeof(double));
    if (result != 0) {
        printf("Problem in posix_memalign for P array. Error code: %d", result);
    }

    initArrays(Nx, Ny, U0, U1, P);

    struct timespec start, finish;
    clock_gettime(CLOCK_REALTIME_COARSE, &start);

    processAlgorithm(Nx, Ny, Nt, Sx, Sy, tao2Hx2, tao2Hy2, tao, U0, U1, P);

    clock_gettime(CLOCK_REALTIME_COARSE, &finish);
    printf("Time: %lf\n", ((double) (finish.tv_sec - start.tv_sec)) + ((double) finish.tv_nsec - start.tv_nsec) / 10E9);

    saveArrayToFile(Nx, Ny, U1, "double00001500.dat");

    result = system("gnuplot gnuplot_script");
    if (result != 0) {
        printf("Problem in system(\"gnuplot gnuplot_script\"). Error code: %d", result);
    }

    free(U0);
    free(U1);
    free(P);

    return 0;
}
 
