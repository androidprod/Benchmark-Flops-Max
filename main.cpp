// main.cpp
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <omp.h>

#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <cstdlib>

/* ===================== OpenCL kernel ===================== */
const char* cl_src = R"(
__kernel void kf(
 __global float* a,
 __global float* b,
 __global float* c,
 int n
){
 int gid = get_global_id(0);
 int stride = get_global_size(0);

 for(int i = gid; i < n; i += stride){
  float x = a[i];
  float y = b[i];

  // 16 FLOPs (FMA)
  x = fma(x,y,y); x = fma(x,y,y);
  x = fma(x,y,y); x = fma(x,y,y);
  x = fma(x,y,y); x = fma(x,y,y);
  x = fma(x,y,y); x = fma(x,y,y);
  x = fma(x,y,y); x = fma(x,y,y);
  x = fma(x,y,y); x = fma(x,y,y);
  x = fma(x,y,y); x = fma(x,y,y);
  x = fma(x,y,y); x = fma(x,y,y);

  c[i] = x;
 }
}
)";

/* ===================== CPU benchmark ===================== */
void run_cpu(double sec){
 const size_t N = 1 << 20; // 1M
 std::vector<float> a(N,1.0f), b(N,1.0f), c(N);

 size_t flops = 0;
 auto st = std::chrono::high_resolution_clock::now();

 for(;;){
  #pragma omp parallel for schedule(static)
  for(size_t i=0;i<N;i++){
   float x = a[i];
   float y = b[i];

   // 8 FLOPs
   x = x*y + y; x = x*y + y;
   x = x*y + y; x = x*y + y;
   x = x*y + y; x = x*y + y;
   x = x*y + y; x = x*y + y;

   c[i] = x;
  }

  flops += N * 8;
  double t = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now()-st).count();

  if(t >= sec){
   std::cout << "CPU MAX\n"
             << (flops/t)/1e9 << " GFLOPS\n";
   break;
  }
 }
}

/* ===================== GPU benchmark ===================== */
void run_gpu(double sec){
 cl_platform_id p; cl_uint pc;
 if(clGetPlatformIDs(1,&p,&pc)!=CL_SUCCESS || pc==0){
  std::cerr<<"No OpenCL platform\n"; return;
 }

 cl_device_id d;
 if(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,nullptr)!=CL_SUCCESS){
  std::cerr<<"GPU not found\n"; return;
 }

 cl_int r;
 cl_context ctx = clCreateContext(nullptr,1,&d,nullptr,nullptr,&r);
 cl_command_queue q = clCreateCommandQueue(ctx,d,0,&r);

 const int N = 1<<20;
 const size_t B = N*sizeof(float);

 cl_mem a = clCreateBuffer(ctx,CL_MEM_READ_ONLY,B,nullptr,&r);
 cl_mem b = clCreateBuffer(ctx,CL_MEM_READ_ONLY,B,nullptr,&r);
 cl_mem c = clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,B,nullptr,&r);

 std::vector<float> h(N,1.0f);
 clEnqueueWriteBuffer(q,a,CL_TRUE,0,B,h.data(),0,nullptr,nullptr);
 clEnqueueWriteBuffer(q,b,CL_TRUE,0,B,h.data(),0,nullptr,nullptr);

 cl_program pr = clCreateProgramWithSource(ctx,1,&cl_src,nullptr,&r);
 clBuildProgram(pr,1,&d,nullptr,nullptr,nullptr);
 cl_kernel k = clCreateKernel(pr,"kf",&r);

 clSetKernelArg(k,0,sizeof(a),&a);
 clSetKernelArg(k,1,sizeof(b),&b);
 clSetKernelArg(k,2,sizeof(c),&c);
 clSetKernelArg(k,3,sizeof(int),&N);

 size_t local = 256;
 size_t global = local * 1024;

 size_t flops = 0;
 auto st = std::chrono::high_resolution_clock::now();

 for(;;){
  clEnqueueNDRangeKernel(q,k,1,nullptr,&global,&local,0,nullptr,nullptr);
  clFinish(q);

  flops += (size_t)N * 16;

  double t = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now()-st).count();

  if(t >= sec){
   std::cout << "GPU MAX\n"
             << (flops/t)/1e9 << " GFLOPS\n";
   break;
  }
 }

 clReleaseKernel(k);
 clReleaseProgram(pr);
 clReleaseMemObject(a);
 clReleaseMemObject(b);
 clReleaseMemObject(c);
 clReleaseCommandQueue(q);
 clReleaseContext(ctx);
}

/* ===================== main ===================== */
int main(int argc,char**argv){
 double sec = 1.0;
 if(argc>1) sec = atof(argv[1]);

 run_cpu(sec);
 run_gpu(sec);
}
