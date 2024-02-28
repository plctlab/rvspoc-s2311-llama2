#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <omp.h>

#define N 300
#define D 32000

#define STRIDE 32
void matmul(int32_t* xout, int32_t* x, int32_t* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    int i;
    register int32_t xi_reg;
    memset(xout, 0, d * sizeof(int32_t));
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
        xi_reg = x[i];
        for (int j = 0; j < d/STRIDE; j++) {
            for(int k = 0; k < STRIDE; k++) {
                xout[j * STRIDE + k] += w[i * d + j * STRIDE + k] * xi_reg;
            }
        }
    }
}

void main()
{
    system("free -m");
    int32_t *w = calloc(N * D, sizeof(int32_t));
    int32_t *x = calloc(N, sizeof(int32_t));
    int32_t *xout = calloc(D, sizeof(int32_t));

    int32_t *p = calloc(8 * N * N, sizeof(int32_t));
    int32_t *w2 = calloc(N * D, sizeof(int32_t));
    int32_t *x2 = calloc(N, sizeof(int32_t));
    int32_t *xout2 = calloc(D, sizeof(int32_t));
    x[0] = 1.0;
    for(int i=0; i<N*D; i++) {
        w[i] = 0.001 * i;
    }
    for(int i=0; i<N; i++)
        x[i] = i;
    
    system("free -m");
    clock_t t1, t2;
    for(int i=0; i<10; i++) {
        t1 = clock();
        for(int j=0; j<8; j++) {
            int32_t *pp = p + j * (N * N);
            matmul(x, x, pp, N, N);
        }

        x2[i] = 1.0;
        matmul(xout, x, w, N, D);
        t2 = clock();
        printf("\n%.3lf ms\n", (double)(t2 - t1)*1000.f / CLOCKS_PER_SEC);
    }

    free(w); free(x); free(xout); free(w2); free(x2); free(xout2);
}