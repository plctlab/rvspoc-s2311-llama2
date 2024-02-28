#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <omp.h>

#define D 32000
#define N 288

// N is small(8,16,32), it speeds up, but not when N is large (128,256).

#include <riscv_vector.h>
#define VLEN 32

void vector_multiply(int n, int d, size_t vlen, const float *w, 
        const float *x, float *xout)
{
    // xout = x[i] * w + xout
    int i;
    vfloat32m8_t vw, vxout;
    vxout = vle32_v_f32m8(xout, vlen);
    for (i=0; i<n; i++) {
        vw = vle32_v_f32m8(w, vlen);
        vxout = vfmacc_vf_f32m8(vxout, x[i], vw, vlen);
        w += d;
    }
    vse32_v_f32m8(xout, vxout, vlen);
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    memset(xout, 0, d * sizeof(float));

    size_t i, j;
    size_t vlen = VLEN;
    // d = 32000, vlen = 32
    const int nn = 4;
	memset(xout, 0, d * sizeof(float));
	clock_t t1,t2,t3, t4;
    #pragma omp parallel for private(i)
	t3 = clock();
	for(j=0; j<n; j+=nn) {
		for(i=0; i<d; i+=vlen) {
			vector_multiply(nn, d, vlen, w+i+j*d, x+j, xout + i);
		}	
	}
}


#define STRIDE 1
void matmulv2(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    int i;
    register float xi_reg;
    memset(xout, 0, d * sizeof(float));
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


void matmulold(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


void main()
{
    system("free -m");
    float *w = calloc(N * D, sizeof(float));
    float *x = calloc(N, sizeof(float));
    float *xout = calloc(D, sizeof(float));
    float *xout2 = calloc(D, sizeof(float));


    for(int i=0; i<N*D; i++) {
        w[i] = i;
    }
    for(int i=0; i<N; i++)
        x[i] = i;
    
    printf("calloc ok.\n");
    clock_t t1, t2, t3, t4;
    double d1=0., d2=0., d3=0.;
    for(int i=0; i<10; i++) {
        t1 = clock();
        matmul(xout, x, w, N, D);
		t2 = clock();
		matmulv2(xout2, x, w, N, D);
        t3 = clock();
		matmulold(xout2, x, w, N, D);
        t4 = clock();
        d1 += (double)(t2 - t1)*1000.f / CLOCKS_PER_SEC;
        d2 += (double)(t3 - t2)*1000.f / CLOCKS_PER_SEC;
        d3 += (double)(t4 - t3)*1000.f / CLOCKS_PER_SEC;
    }

    printf("\n%.3lf  %.3lf  %.3lf ms\n", d1/10, d3/10, d1/d3);

    for(int i=0; i<20; i+=5)
        printf("%.2f  %.2f\n", xout[i], xout2[i]);

    free(w); free(x); free(xout);
}
