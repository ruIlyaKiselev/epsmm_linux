#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define Xa 0.0f
#define Xb 4.0f
#define Ya 0.0f
#define Yb 4.0f

#define Sx  1
#define Sy  1

#define F0  1.0f
#define T0  1.5f

#define Y 4.0f

//#define FTYPE double
//#define VECTOR_SIZE 32
//typedef FTYPE vec __attribute__ ((vector_size (VECTOR_SIZE)))

int counter = 0;

static void
count_line(const double *U_cur, double *U_prev, const double *P, int count, int ld, double h2x2t2, double h2y2t2) {
    register const double *end = U_cur + count;
    __asm__ __volatile__(".intel_syntax noprefix\n\t"
                         "mov eax, %9\n\t"
                         "sub rax, 4\n\t"
                         "shl rax, 3\n\t"

                         "mov rbx, %1\n\t"
                         "add rbx, rax\n\t"

                         "mov eax, %3\n\t"
                         "shl rax, 3\n\t"

                         "jmp .end_cycle%=\n\t"
                         ".cycle%=:\n\t"
                         "vmovupd ymm0, [%1 - 8]\n\t"      //U_left
                         "vmovupd ymm1, [%1]\n\t"           //U_cur
                         "vmovupd ymm2, [%1+8]\n\t"     //U_right

                         "vmovupd ymm3, [%1 + rax]\n\t"     //U_low
                         "mov rcx, %1\n\t"
                         "sub rcx, rax\n\t"
                         "vmovupd ymm4, [rcx]\n\t"     //U_top

                         "vmovupd ymm5, [%2]\n\t"     //P_cur
                         "mov rcx, %2\n\t"
                         "sub rcx, rax\n\t"
                         "vmovupd ymm6, [rcx]\n\t"     //P_top
                         "vmovupd ymm7, [%2 - 8]\n\t"     //P_left
                         "vmovupd ymm8, [rcx - 8]\n\t"     //P_top_left

                         "vsubpd ymm10, ymm2, ymm1\n\t"  //U_right - U_cur
                         "vaddpd ymm11, ymm6, ymm5\n\t" //P_top + P_cur
                         "vmulpd ymm10, ymm10, ymm11\n\t" //(U_right - U_cur) * (P_top + P_cur)

                         "vsubpd ymm11, ymm0, ymm1\n\t"  //U_left - U_cur
                         "vaddpd ymm12, ymm8, ymm7\n\t" //P_top_left + P_left
                         "vfmadd231pd ymm10, ymm11, ymm12\n\t" //(U_left - U_cur) * (P_top_left + P_cur) + (U_right - U_cur) * (P_top + P_cur)
                         "vmovddup xmm9, %7\n\t"
                         "vinsertf128 ymm9, ymm9, xmm9, 0x1\n\t"
                         "vmulpd ymm10, ymm10, ymm9\n\t" //Previous result * h2x2t2   (1)

                         "vsubpd ymm11, ymm3, ymm1\n\t"  //U_low - U_cur
                         "vaddpd ymm12, ymm7, ymm5\n\t" //P_left + P_cur
                         "vmulpd ymm11, ymm11, ymm12\n\t" //(U_low - U_cur) * (P_left + P_cur)

                         "vsubpd ymm12, ymm4, ymm1\n\t"  //U_top - U_cur
                         "vaddpd ymm13, ymm8, ymm6\n\t" //P_top_left + P_top
                         "vfmadd231pd ymm11, ymm12, ymm13\n\t" //(U_top - U_cur) * (P_top_left + P_top) + (U_low - U_cur) * (P_left + P_cur)
                         "vmovddup xmm9, %8\n\t"
                         "vinsertf128 ymm9, ymm9, xmm9, 0x1\n\t"
                         "vfmadd231pd ymm10, ymm11, ymm9\n\t"  //(1) + previous result * h2y2t2

                         "vmovupd ymm9, [%0]\n\t"      //U_prev

                         "vaddpd ymm11, ymm1, ymm1\n\t"  //2*U_cur
                         "vsubpd ymm11, ymm11, ymm9\n\t"
                         "vaddpd ymm10, ymm11, ymm10\n\t"
                         "vmovupd [%0], ymm10\n\r"

                         "add %0, 32\n\t"
                         "add %1, 32\n\t"
                         "add %2, 32\n\t"

                         ".end_cycle%=:\n\t"
                         "cmp %1, rbx\n\t"
                         "jle .cycle%=\n\t"

                         ".att_syntax \n\t"
    : "=r"(U_prev), "=r"(U_cur), "=r"(P)
    :"r"(ld), "0"(U_prev), "1"(U_cur), "2"(P), "x"(h2x2t2), "x"(h2y2t2), "r"(count)
    : "rax", "rbx", "rcx", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13"
    );

    while (U_cur != end) {
        register double Ucur;
        Ucur = U_cur[0];
        *U_prev = 2 * Ucur - *U_prev +
                  ((U_cur[1] - Ucur) * (P[-ld] + P[0]) +
                   (U_cur[-1] - Ucur) * (P[-ld - 1] + P[-1])) * h2x2t2 +
                  ((U_cur[ld] - Ucur) * (P[-1] + P[0]) +
                   (U_cur[-ld] - Ucur) * (P[-ld - 1] + P[-ld])) * h2y2t2;
        U_prev++;
        U_cur++;
        P++;
    }
}

void wave(int Nx, int Ny, int Nt) {
    FILE *fp;
    const double t = Nx <= 1000 && Ny <= 1000 ? 0.01f : 0.001f;
    register const double h2x2t2 = t * t / (2 * (Xb - Xa) / (double) (Nx - 1) * (Xb - Xa) / (Nx - 1));
    register const double h2y2t2 = t * t / (2 * (Yb - Ya) / (double) (Ny - 1) * (Yb - Ya) / (Nx - 1));

    double *U_prev = (double *) malloc(sizeof(double) * Nx * Ny);
    double *U_cur = (double *) malloc(sizeof(double) * Nx * Ny);
    double *P = (double *) malloc(sizeof(double) * Nx * Ny);


    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            U_cur[i * Nx + j] = 0.0f;
            U_prev[i * Nx + j] = 0.0f;
            P[i * Nx + j] = j < Nx / 2 ? 0.01 : 0.04;
        }
    }

#pragma omp parallel
    {

        int count = (Nx - 2) / omp_get_num_threads() + ((Nx - 2) % omp_get_num_threads() > omp_get_thread_num());
        int shift =
                ((Nx - 2) / omp_get_num_threads()) * omp_get_thread_num() +
                ((Nx - 2) % omp_get_num_threads() > omp_get_thread_num() ? omp_get_thread_num() :
                 (Nx - 2) % omp_get_num_threads());

        for (int T = 0; T < Nt; T++) {
            double *tmpUcur = U_cur + shift + Nx + 1;
            double *tmpUprev = U_prev + shift + Nx + 1;
            double *tmpP = P + shift + Nx + 1;
            double save = U_prev[Sy * Nx + Sx];

            for (int i = 1; i < Nx - 1; ++i) {
                count_line(tmpUcur, tmpUprev, tmpP, count, Nx, h2x2t2, h2y2t2);
                tmpUcur += Nx;
                tmpUprev += Nx;
                tmpP += Nx;
            }
#pragma omp barrier
            if (shift <= Sx && Sx < count + shift) {

                double ft = exp(-(2 * M_PI * F0 * (T * t - T0)) * (2 * M_PI * F0 * (T * t - T0)) / (Y * Y)) *
                            sin(2 * M_PI * F0 * (T * t - T0)) * t * t;
                register double Ucur;

                Ucur = U_cur[Sy * Nx + Sx];
                U_prev[Sy * Nx + Sx] = 2 * Ucur - save +
                                       ((U_cur[Sy * Nx + Sx + 1] - Ucur) *
                                        (P[(Sy - 1) * Nx + Sx] + P[Sy * Nx + Sx]) +
                                        (U_cur[Sy * Nx + Sx - 1] - Ucur) *
                                        (P[(Sy - 1) * Nx + Sx - 1] + P[Sy * Nx + Sx - 1])) *
                                       h2x2t2 +
                                       ((U_cur[(Sy + 1) * Nx + Sx] - Ucur) *
                                        (P[Sy * Nx + Sx - 1] + P[Sy * Nx + Sx]) +
                                        (U_cur[(Sy - 1) * Nx + Sx] - Ucur) *
                                        (P[(Sy - 1) * Nx + Sx - 1] + P[(Sy - 1) * Nx + Sx])) *
                                       h2y2t2 + ft;

                printf("%d %lf\n", counter, ft);

                double *tmp = U_cur;
                U_cur = U_prev;
                U_prev = tmp;
                counter += 1;
            }
#pragma omp barrier
        }
    }

    fp = fopen("../new.dat", "wb");
    fwrite(U_cur, sizeof(double), Nx * Ny, fp);
    fclose(fp);
    free(U_cur);
    free(U_prev);
    free(P);
}

int main(int argc, char **argv) {
    int Nx;
    int Ny;
    int Nt;
    char *end;
    if (argc < 5 || argc > 5) {
        fprintf(stderr, "Wrong count of arguments\n");
        return 0;
    }

    Nx = (int) strtol(argv[1], &end, 10);
    Ny = (int) strtol(argv[2], &end, 10);
    Nt = (int) strtol(argv[3], &end, 10);
    omp_set_num_threads(strtol(argv[4], &end, 10));

    struct timespec start, endt;

    clock_gettime(CLOCK_REALTIME_COARSE, &start);

    wave(Nx, Ny, Nt);

    clock_gettime(CLOCK_REALTIME_COARSE, &endt);

    printf("Sec: %f\n", (double) endt.tv_sec - start.tv_sec + ((double) endt.tv_nsec - start.tv_nsec) / 1000000000);

    return 0;
}
 
