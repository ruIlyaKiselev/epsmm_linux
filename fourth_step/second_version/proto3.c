#include <stdio.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

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

void processLine(int Nx, int Ny, int Sx, int Sy, int i, int shift, int count, double tao2Hx2, double tao2Hy2, double tao2functionF, double* U0, double* U1, double* P) {

    double *pU0 = U0 + Nx * i + shift;
    double *pU1 = U1 + Nx * i + shift;
    double *pP = P + Nx * i + shift;
    double *end = pU1 + count;

    int j;
    for (j = 1; j <= Nx - 4 - 1; j += 4) {
        algorithmStepVector(Nx, Ny, tao2Hx2, tao2Hy2, pU0, pU1, pP);

        pU0 += 4;
        pU1 += 4;
        pP += 4;
    }

    while (pU1 != end) {
        algorithmStepScalar(Nx, Ny, tao2Hx2, tao2Hy2, pU0, pU1, pP);
        pU0++;
        pU1++;
        pP++;
        j++;
    }

    if (i == Sy && shift <= Sx && Sx < shift + count) {
        finalStep(Nx, Ny, Sx, Sy, tao2functionF, tao2Hx2, tao2Hy2, U0, U1, P);
    }

    pU0 += 2;
    pU1 += 2;
    pP += 2;
}

void greenStep(int Nx, int Ny, int Sx, int Sy, int shift, int count, int i, double tao2Hx2, double tao2Hy2, double t2f1, double t2f2, double t2f3, double t2f4, double *U0, double *U1, double *P) {
    processLine(Nx, Ny, Sx, Sy, i - 1, shift - 1 < 1 ? 1 : shift - 1, shift - 1 < 1 ? 1 : 2, tao2Hx2, tao2Hy2, t2f2, U0, U1, P);
    processLine(Nx, Ny, Sx, Sy, i - 2, shift - 2 < 1 ? 1 : shift - 2, shift - 2 < 1 ? 2 : 4, tao2Hx2, tao2Hy2, t2f3, U1, U0, P);
    processLine(Nx, Ny, Sx, Sy, i - 3, shift - 3 < 1 ? 1 : shift - 3, shift - 3 < 1 ? 3 : 6, tao2Hx2, tao2Hy2, t2f4, U0, U1, P);
}

void redStep(int Nx, int Ny, int Sx, int Sy, int shift, int count, int i, double tao2Hx2, double tao2Hy2, double t2f1, double t2f2, double t2f3, double t2f4, double *U0, double *U1, double *P) {
    if (omp_get_thread_num() == omp_get_num_threads() - 1) {
        count = count - 3;
    } else {
        count = count - 6;
    }

    processLine(Nx, Ny, Sx, Sy, i, shift, count, tao2Hx2, tao2Hy2, t2f1, U1, U0, P);
    processLine(Nx, Ny, Sx, Sy, i - 1, shift + 1, count, tao2Hx2, tao2Hy2, t2f2, U0, U1, P);
    processLine(Nx, Ny, Sx, Sy, i - 2, shift + 2, count, tao2Hx2, tao2Hy2, t2f3, U1, U0, P);
    processLine(Nx, Ny, Sx, Sy, i - 3, shift + 3, count, tao2Hx2, tao2Hy2, t2f4, U0, U1, P);
}

void blueStep(int Nx, int Ny, int Sx, int Sy, int shift, int count, int i, double tao2Hx2, double tao2Hy2, double t2f1, double t2f2, double t2f3, double t2f4, double *U0, double *U1, double *P) {
    if (omp_get_thread_num() == omp_get_num_threads() - 1) {
        shift = shift + count - 3;
        count = 3;
    } else {
        shift = shift + count - 6;
        count = 6;
    }

    processLine(Nx, Ny, Sx, Sy, i, shift, count, tao2Hx2, tao2Hy2, t2f1, U1, U0, P);
    processLine(Nx, Ny, Sx, Sy, i - 1, shift + 1, count % 2 ? count - 1 : count - 2, tao2Hx2, tao2Hy2, t2f2, U0, U1, P);
    processLine(Nx, Ny, Sx, Sy, i - 2, shift + 2, count % 2 ? count - 2 : count - 4, tao2Hx2, tao2Hy2, t2f3, U1, U0, P);
}

void processAlgorithm(int Nx, int Ny, int Nt, int Sx, int Sy, double tao2Hx2, double tao2Hy2, double tao, double* U0, double* U1, double* P) {

#pragma omp parallel proc_bind(close)
    {
        int count = (Nx - 2) / omp_get_num_threads() + ((Nx - 2) % omp_get_num_threads() > omp_get_thread_num());
        int shift = ((Nx - 2) / omp_get_num_threads()) * omp_get_thread_num() +
                ((Nx - 2) % omp_get_num_threads() > omp_get_thread_num() ? omp_get_thread_num() : (Nx - 2) % omp_get_num_threads()) + 1;

        for (int n = 0; n < Nt; n += 4) {

            double tao2functionF1 = tao * tao * functionF(n, Nx, Ny);
            double tao2functionF2 = tao * tao * functionF(n + 1, Nx, Ny);
            double tao2functionF3 = tao * tao * functionF(n + 3, Nx, Ny);
            double tao2functionF4 = tao * tao * functionF(n + 4, Nx, Ny);

            processLine (Nx, Ny, Sx, Sy, 1, shift, count, tao2Hx2, tao2Hy2, tao2functionF1, U1, U0, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, 2, shift, count, tao2Hx2, tao2Hy2, tao2functionF1, U1, U0, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, 1, shift, count, tao2Hx2, tao2Hy2, tao2functionF2, U0, U1, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, 3, shift, count, tao2Hx2, tao2Hy2, tao2functionF1, U1, U0, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, 2, shift, count, tao2Hx2, tao2Hy2, tao2functionF2, U0, U1, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, 1, shift, count, tao2Hx2, tao2Hy2, tao2functionF3, U1, U0, P);
#pragma omp barrier

            for (int i = 4; i < Ny - 1; i++) {
                greenStep(Nx, Ny, Sx, Sy, shift, count, i, tao2Hx2, tao2Hy2,
                          tao2functionF1, tao2functionF2, tao2functionF3, tao2functionF4,
                          U1, U0, P);
#pragma omp barrier
                redStep(Nx, Ny, Sx, Sy, shift, count, i, tao2Hx2, tao2Hy2,
                          tao2functionF1, tao2functionF2, tao2functionF3, tao2functionF4,
                          U1, U0, P);
#pragma omp barrier
                blueStep(Nx, Ny, Sx, Sy, shift, count, i, tao2Hx2, tao2Hy2,
                        tao2functionF1, tao2functionF2, tao2functionF3, tao2functionF4,
                        U1, U0, P);
#pragma omp barrier
            }

            processLine (Nx, Ny, Sx, Sy, Ny - 2, shift, count, tao2Hx2, tao2Hy2, tao2functionF2, U0, U1, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, Ny - 3, shift, count, tao2Hx2, tao2Hy2, tao2functionF3, U1, U0, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, Ny - 4, shift, count, tao2Hx2, tao2Hy2, tao2functionF4, U0, U1, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, Ny - 2, shift, count, tao2Hx2, tao2Hy2, tao2functionF3, U1, U0, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, Ny - 3, shift, count, tao2Hx2, tao2Hy2, tao2functionF4, U0, U1, P);
#pragma omp barrier
            processLine (Nx, Ny, Sx, Sy, Ny - 2, shift, count, tao2Hx2, tao2Hy2, tao2functionF4, U0, U1, P);
#pragma omp barrier
        }
    }
}

int main(int argc, char **argv) {
    int Nx;
    int Ny;
    int Nt;

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

    U0 = (double*)_mm_malloc(Nx * Ny * sizeof (double), 32);
    U1 = (double*)_mm_malloc(Nx * Ny * sizeof (double), 32);
    P = (double*)_mm_malloc(Nx * Ny * sizeof (double), 32);

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

    _mm_free(U0);
    _mm_free(U1);
    _mm_free(P);

    return 0;
}
 
