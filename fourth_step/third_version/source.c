#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <semaphore.h>

#define Xa 0.0f
#define Xb 4.0f
#define Ya 0.0f
#define Yb 4.0f

#define Sx  1
#define Sy  1

#define F0  1.0f
#define T0  1.5f

#define Y 4.0f

void
count_line(double *U_cur, double *U_prev, double *P, int ld, int shift, int count, int i, double h2x2t2, double h2y2t2,
           double ft) {
    register double *tmpPrev = U_prev + ld * i + shift;
    register double *tmpCur = U_cur + ld * i + shift;
    register double *tmpP = P + ld * i + shift;
    register double Ucur;
    register double *endCur = tmpCur + count;
    double save = U_prev[Sy * ld + Sx];

    __asm__ __volatile__(".intel_syntax noprefix\n\t"
                         "mov eax, %9\n\t"
                         "sub rax, 3\n\t"
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
                         "jl .cycle%=\n\t"

                         ".att_syntax \n\t"
    : "=r"(tmpPrev), "=r"(tmpCur), "=r"(tmpP)
    :"r"(ld), "0"(tmpPrev), "1"(tmpCur), "2"(tmpP), "x"(h2x2t2), "x"(h2y2t2), "r"(count)
    : "rax", "rbx", "rcx", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13"
    );

    while (tmpCur != endCur) {
        Ucur = tmpCur[0];
        *tmpPrev = 2 * Ucur - *tmpPrev +
                   ((tmpCur[1] - Ucur) * (tmpP[-ld] + tmpP[0]) +
                    (tmpCur[-1] - Ucur) * (tmpP[-ld - 1] + tmpP[-1])) * h2x2t2 +
                   ((tmpCur[ld] - Ucur) * (tmpP[-1] + tmpP[0]) +
                    (tmpCur[-ld] - Ucur) * (tmpP[-ld - 1] + tmpP[-ld])) * h2y2t2;
        tmpPrev++;
        tmpCur++;
        tmpP++;
    }

    if (i == Sy && shift <= Sx && Sx < shift + count) {

        Ucur = U_cur[Sy * ld + Sx];
        U_prev[Sy * ld + Sx] = 2 * Ucur - save +
                               ((U_cur[Sy * ld + Sx + 1] - Ucur) *
                                (P[(Sy - 1) * ld + Sx] + P[Sy * ld + Sx]) +
                                (U_cur[Sy * ld + Sx - 1] - Ucur) *
                                (P[(Sy - 1) * ld + Sx - 1] + P[Sy * ld + Sx - 1])) *
                               h2x2t2 +
                               ((U_cur[(Sy + 1) * ld + Sx] - Ucur) *
                                (P[Sy * ld + Sx - 1] + P[Sy * ld + Sx]) +
                                (U_cur[(Sy - 1) * ld + Sx] - Ucur) *
                                (P[(Sy - 1) * ld + Sx - 1] + P[(Sy - 1) * ld + Sx])) *
                               h2y2t2 + ft;
    }
}

void step_initialize(double *U_cur, double *U_prev, double *P, int ld, int shift, int i, double h2x2t2,
                     double h2y2t2, double ft2, double ft3, double ft4) {

    count_line(U_prev, U_cur, P, ld, shift - 1 < 1 ? 1 : shift - 1, shift - 1 < 1 ? 1 : 2, i - 1, h2x2t2,
               h2y2t2, ft2);
    count_line(U_cur, U_prev, P, ld, shift - 2 < 1 ? 1 : shift - 2, shift - 2 < 1 ? 2 : 4, i - 2, h2x2t2,
               h2y2t2, ft3);
    count_line(U_prev, U_cur, P, ld, shift - 3 < 1 ? 1 : shift - 3, shift - 3 < 1 ? 3 : 6, i - 3, h2x2t2,
               h2y2t2, ft4);
}

void
step_main(double *U_cur, double *U_prev, double *P, int ld, int shift, int count, int i, double h2x2t2,
          double h2y2t2, double ft1, double ft2, double ft3, double ft4) {
    if (omp_get_thread_num() == omp_get_num_threads() - 1) {
        count = count - 3;
    } else {
        count = count - 6;
    }
    count_line(U_cur, U_prev, P, ld, shift, count, i, h2x2t2, h2y2t2, ft1);
    count_line(U_prev, U_cur, P, ld, shift + 1, count, i - 1, h2x2t2, h2y2t2, ft2);
    count_line(U_cur, U_prev, P, ld, shift + 2, count, i - 2, h2x2t2, h2y2t2, ft3);
    count_line(U_prev, U_cur, P, ld, shift + 3, count, i - 3, h2x2t2, h2y2t2, ft4);
}

void
step_final(double *U_cur, double *U_prev, double *P, int ld, int shift, int count, int i, double h2x2t2, double h2y2t2,
           double ft1, double ft2, double ft3) {
    if (omp_get_thread_num() == omp_get_num_threads() - 1) {
        shift = shift + count - 3;
        count = 3;
    } else {
        shift = shift + count - 6;
        count = 6;
    }
    count_line(U_cur, U_prev, P, ld, shift, count, i, h2x2t2, h2y2t2, ft1);
    count_line(U_prev, U_cur, P, ld, shift + 1, count % 2 ? count - 1 : count - 2, i - 1, h2x2t2, h2y2t2, ft2);
    count_line(U_cur, U_prev, P, ld, shift + 2, count % 2 ? count - 2 : count - 4, i - 2, h2x2t2, h2y2t2, ft3);
}

void wave(int Nx, int Ny, int Nt, int countThread) {
    FILE *fp;
    const double t = Nx <= 1000 && Ny <= 1000 ? 0.01f : 0.001f;
    register const double h2x2t2 = t * t / (2 * (Xb - Xa) / (double) (Nx - 1) * (Xb - Xa) / (Nx - 1));
    register const double h2y2t2 = t * t / (2 * (Yb - Ya) / (double) (Ny - 1) * (Yb - Ya) / (Nx - 1));

    double *U_prev = (double *) malloc(sizeof(double) * Nx * Ny);
    double *U_cur = (double *) malloc(sizeof(double) * Nx * Ny);
    double *P = (double *) malloc(sizeof(double) * Nx * Ny);
    sem_t *semaphore_gr = malloc(sizeof(sem_t) * countThread);
    sem_t *semaphore_blue = malloc(sizeof(sem_t) * countThread);

    for (int i = 0; i < countThread; ++i) {
        sem_init(semaphore_gr + i, 0, 1);
        sem_init(semaphore_blue + i, 0, 0);
    }

    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            U_cur[i * Nx + j] = 0.0f;
            U_prev[i * Nx + j] = 0.0f;
            P[i * Nx + j] = 0.01;
        }
    }
#pragma omp parallel
    {
        int count = (Nx - 2) / omp_get_num_threads() + ((Nx - 2) % omp_get_num_threads() > omp_get_thread_num());
        int shift =
                ((Nx - 2) / omp_get_num_threads()) * omp_get_thread_num() +
                ((Nx - 2) % omp_get_num_threads() > omp_get_thread_num() ? omp_get_thread_num() :
                 (Nx - 2) % omp_get_num_threads()) + 1;
        for (int T = 0; T < Nt; T += 4) {
            double ft1 = exp(-(2 * M_PI * F0 * (T * t - T0)) * (2 * M_PI * F0 * (T * t - T0)) / (Y * Y)) *
                         sin(2 * M_PI * F0 * (T * t - T0)) * t * t;
            double ft2 = exp(-(2 * M_PI * F0 * ((T + 1) * t - T0)) * (2 * M_PI * F0 * ((T + 1) * t - T0)) / (Y * Y)) *
                         sin(2 * M_PI * F0 * ((T + 1) * t - T0)) * t * t;
            double ft3 = exp(-(2 * M_PI * F0 * ((T + 2) * t - T0)) * (2 * M_PI * F0 * ((T + 2) * t - T0)) / (Y * Y)) *
                         sin(2 * M_PI * F0 * ((T + 2) * t - T0)) * t * t;
            double ft4 = exp(-(2 * M_PI * F0 * ((T + 3) * t - T0)) * (2 * M_PI * F0 * ((T + 3) * t - T0)) / (Y * Y)) *
                         sin(2 * M_PI * F0 * ((T + 3) * t - T0)) * t * t;

            count_line(U_cur, U_prev, P, Nx, shift, count, 1, h2x2t2, h2y2t2, ft1);
#pragma omp barrier
            count_line(U_cur, U_prev, P, Nx, shift, count, 2, h2x2t2, h2y2t2, ft1);
#pragma omp barrier
            count_line(U_prev, U_cur, P, Nx, shift, count, 1, h2x2t2, h2y2t2, ft2);
#pragma omp barrier
            count_line(U_cur, U_prev, P, Nx, shift, count, 3, h2x2t2, h2y2t2, ft1);
#pragma omp barrier
            count_line(U_prev, U_cur, P, Nx, shift, count, 2, h2x2t2, h2y2t2, ft2);
#pragma omp barrier
            count_line(U_cur, U_prev, P, Nx, shift, count, 1, h2x2t2, h2y2t2, ft3);
#pragma omp barrier

            for (int i = 4; i < Ny - 1; ++i) {
                if (omp_get_thread_num() != 0)
                    sem_wait(semaphore_gr + omp_get_thread_num());
                step_initialize(U_cur, U_prev, P, Nx, shift, i, h2x2t2, h2y2t2, ft2, ft3, ft4);
                if (omp_get_thread_num() != 0)
                    sem_post(semaphore_blue + omp_get_thread_num() - 1);
                step_main(U_cur, U_prev, P, Nx, shift, count, i, h2x2t2, h2y2t2, ft1, ft2, ft3, ft4);
                if (omp_get_thread_num() != countThread - 1)
                    sem_wait(semaphore_blue + omp_get_thread_num());
                step_final(U_cur, U_prev, P, Nx, shift, count, i, h2x2t2, h2y2t2, ft1, ft2, ft3);
                if (omp_get_thread_num() != countThread - 1)
                    sem_post(semaphore_gr + omp_get_thread_num() + 1);
            }

            count_line(U_prev, U_cur, P, Nx, shift, count, Ny - 2, h2x2t2, h2y2t2, ft2);
#pragma omp barrier
            count_line(U_cur, U_prev, P, Nx, shift, count, Ny - 3, h2x2t2, h2y2t2, ft3);
#pragma omp barrier
            count_line(U_prev, U_cur, P, Nx, shift, count, Ny - 4, h2x2t2, h2y2t2, ft4);
#pragma omp barrier
            count_line(U_cur, U_prev, P, Nx, shift, count, Ny - 2, h2x2t2, h2y2t2, ft3);
#pragma omp barrier
            count_line(U_prev, U_cur, P, Nx, shift, count, Ny - 3, h2x2t2, h2y2t2, ft4);
#pragma omp barrier
            count_line(U_prev, U_cur, P, Nx, shift, count, Ny - 2, h2x2t2, h2y2t2, ft4);
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
    int countThread;
    char *end;

    if (argc < 5 || argc > 5) {
        fprintf(stderr, "Wrong count of arguments\n");
        return 0;
    }

    Nx = (int) strtol(argv[1], &end, 10);
    Ny = (int) strtol(argv[2], &end, 10);
    Nt = (int) strtol(argv[3], &end, 10);
    countThread = strtol(argv[4], &end, 10);
    omp_set_num_threads(countThread);

    struct timespec start, endt;

    clock_gettime(CLOCK_REALTIME_COARSE, &start);

    wave(Nx, Ny, Nt, countThread);

    clock_gettime(CLOCK_REALTIME_COARSE, &endt);

    printf("Sec: %f\n", (double) endt.tv_sec - start.tv_sec + ((double) endt.tv_nsec - start.tv_nsec) / 1000000000);

    return 0;
} 
