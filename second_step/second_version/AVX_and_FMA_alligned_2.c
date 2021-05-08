#include <stdio.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

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

    __m256d leftU1 = _mm256_load_pd(pU1 - 1);
    __m256d rightU1 = _mm256_load_pd(pU1 - 1 + 4);

    __m256d leftTopU1 = _mm256_load_pd(pU1 - 1 - Nx);
    __m256d rightTopU1 = _mm256_load_pd(pU1 - 1 + 4 - Nx);

    __m256d leftBottomU1 = _mm256_load_pd(pU1 - 1 + Nx);
    __m256d rightBottomU1 = _mm256_load_pd(pU1 - 1 + 4 + Nx);

    __m256d leftP = _mm256_load_pd(pP - 1);
    __m256d rightP = _mm256_load_pd(pP - 1 + 4);

    __m256d leftTopP = _mm256_load_pd(pP - 1 - Nx);
    __m256d rightTopP = _mm256_load_pd(pP - 1 + 4 - Nx);

    __m256d U1ijm1V = leftU1;

    __m256d leftPartU1ijV = _mm256_permute4x64_pd(leftU1, 57);
    __m256d rightPartU1ijV = _mm256_permute4x64_pd(rightU1, 57);
    __m256d U1ijV = _mm256_blend_pd(leftPartU1ijV, rightPartU1ijV, 8);

    __m256d U1ijp1V = _mm256_permute2f128_pd(leftU1, rightU1, 33);

    __m256d leftTopPartU1ijV = _mm256_permute4x64_pd(leftTopU1, 57);
    __m256d rightTopPartU1ijV = _mm256_permute4x64_pd(rightTopU1, 57);
    __m256d U1im1jV = _mm256_blend_pd(leftTopPartU1ijV, rightTopPartU1ijV, 8);

    __m256d leftBottomPartU1ijV = _mm256_permute4x64_pd(leftBottomU1, 57);
    __m256d rightBottomPartU1ijV = _mm256_permute4x64_pd(rightBottomU1, 57);
    __m256d U1ip1jV = _mm256_blend_pd(leftBottomPartU1ijV, rightBottomPartU1ijV, 8);

    __m256d leftPartPijV = _mm256_permute4x64_pd(leftP, 57);
    __m256d rightPartPijV = _mm256_permute4x64_pd(rightP, 57);
    __m256d PijV = _mm256_blend_pd(leftPartPijV, rightPartPijV, 8);

    __m256d Pim1jm1V = leftTopP;

    __m256d leftPartPim1jV = _mm256_permute4x64_pd(leftTopP, 57);
    __m256d rightPartPim1jV = _mm256_permute4x64_pd(rightTopP, 57);
    __m256d Pim1jV = _mm256_blend_pd(leftPartPim1jV, rightPartPim1jV, 8);

    __m256d Pijm1V = leftP;

    __m256d leftPartX = _mm256_mul_pd(_mm256_sub_pd(U1ijp1V, U1ijV), _mm256_add_pd(Pim1jV, PijV));
    __m256d rightPartX = _mm256_mul_pd(_mm256_sub_pd(U1ijm1V, U1ijV), _mm256_add_pd(Pim1jm1V, Pijm1V));
    __m256d leftPartY = _mm256_mul_pd(_mm256_sub_pd(U1ip1jV, U1ijV), _mm256_add_pd(Pijm1V, PijV));
    __m256d rightPartY = _mm256_mul_pd(_mm256_sub_pd(U1im1jV, U1ijV),_mm256_add_pd(Pim1jm1V,Pim1jV));

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

void processAlgorithm(int Nx, int Ny, int Nt, int Sx, int Sy, double tao2Hx2, double tao2Hy2, double tao, double* U0, double* U1, double* P) {

    for (int n = 0; n != Nt; n++) {
        double tao2functionF = tao * tao * functionF(n, Nx, Ny);

        double *pU0 = U0 + Nx + 1;
        double *pU1 = U1 + Nx + 1;
        double *pP = P + Nx + 1;

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

        double *tmp = U1;
        U1 = U0;
        U0 = tmp;
    }
}

int main(int argc, char **argv) {
    int Nx = 10000, Ny = 10000;
    int Nt = 100;

    if (argc == 4) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nt = atoi(argv[3]);
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

//    double* U;
//    U = (double*)_mm_malloc(100 * sizeof (double), 32);
//
//    for (int i = 0; i != 100; i++) {
//        U[i] = i;
//    }
//
//    __m256d left = _mm256_load_pd(U);
//    __m256d center = _mm256_load_pd(U + 4);
//    __m256d right = _mm256_load_pd(U + 8);
//
//    __m256d leftPartUm1 = _mm256_permute4x64_pd(left, 3);
//    __m256d rightPartUm1 = _mm256_permute4x64_pd(center, 144);
//    __m256d Um1 = _mm256_blend_pd(leftPartUm1, rightPartUm1, 14);
//
//    __m256d leftPartUp1 = _mm256_permute4x64_pd(center, 57);
//    __m256d rightPartUp1 = _mm256_permute4x64_pd(right, 3);
//    __m256d Up1 = _mm256_blend_pd(leftPartUp1, rightPartUp1, 8);
//
//    __m256d leftPartU1 = _mm256_permute4x64_pd(center, 57);
//    __m256d rightPartU1 = _mm256_permute4x64_pd(right, 57);
//    __m256d U1 = _mm256_blend_pd(leftPartU1, rightPartU1, 8);
//
//    __m256d U1p1 = _mm256_permute2f128_pd(center, right, 33);
//
//    double *sLeft = (double *) &left;
//    double *sCenter = (double *) &center;
//    double *sRight = (double *) &right;
//
//    double *sLeftPartUm1 = (double *) &leftPartUm1;
//    double *sRightPartUm1 = (double *) &rightPartUm1;
//    double *sUm1 = (double *) &Um1;
//
//    double *sLeftPartUp1 = (double *) &leftPartUp1;
//    double *sRightPartUp1 = (double *) &rightPartUp1;
//    double *sUp1 = (double *) &Up1;
//
//    double *sLeftPartU1 = (double *) &leftPartU1;
//    double *sRightPartU1 = (double *) &rightPartU1;
//    double *sU1 = (double *) &U1;
//
//    double *sU1p1 = (double *) &U1p1;
//
//    printf("left:         %d %d %d %d\n", (int) sLeft[0], (int) sLeft[1], (int) sLeft[2], (int) sLeft[3]);
//    printf("center:       %d %d %d %d\n", (int) sCenter[0], (int) sCenter[1], (int) sCenter[2], (int) sCenter[3]);
//    printf("right:        %d %d %d %d\n\n", (int) sRight[0], (int) sRight[1], (int) sRight[2], (int) sRight[3]);
//
//    printf("leftPartUm1:  %d %d %d %d\n", (int) sLeftPartUm1[0], (int) sLeftPartUm1[1], (int) sLeftPartUm1[2], (int) sLeftPartUm1[3]);
//    printf("rightPartUm1: %d %d %d %d\n", (int) sRightPartUm1[0], (int) sRightPartUm1[1], (int) sRightPartUm1[2], (int) sRightPartUm1[3]);
//    printf("Um1:          %d %d %d %d\n\n", (int) sUm1[0], (int) sUm1[1], (int) sUm1[2], (int) sUm1[3]);
//
//    printf("leftPartUp1:  %d %d %d %d\n", (int) sLeftPartUp1[0], (int) sLeftPartUp1[1], (int) sLeftPartUp1[2], (int) sLeftPartUp1[3]);
//    printf("rightPartUp1: %d %d %d %d\n", (int) sRightPartUp1[0], (int) sRightPartUp1[1], (int) sRightPartUp1[2], (int) sRightPartUp1[3]);
//    printf("Up1:          %d %d %d %d\n\n", (int) sUp1[0], (int) sUp1[1], (int) sUp1[2], (int) sUp1[3]);
//
//    printf("leftPartU1:   %d %d %d %d\n", (int) sLeftPartU1[0], (int) sLeftPartU1[1], (int) sLeftPartU1[2], (int) sLeftPartU1[3]);
//    printf("rightPartU1:  %d %d %d %d\n", (int) sRightPartU1[0], (int) sRightPartU1[1], (int) sRightPartU1[2], (int) sRightPartU1[3]);
//    printf("U1:           %d %d %d %d\n\n", (int) sU1[0], (int) sU1[1], (int) sU1[2], (int) sU1[3]);
//
//    printf("U1p1:         %d %d %d %d\n", (int) sU1p1[0], (int) sU1p1[1], (int) sU1p1[2], (int) sU1p1[3]);

    return 0;
}
 
