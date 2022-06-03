/*
 ============================================================================
 Original Author  : G. Barlas
 Added Authors    : Saúl Contreras
                  : Michele Benvenuto
 Version          : 1.1
 Last modified    : May 2022
 License          : Released under the GNU GPL 3.0
 Description      :
 To build use     : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// Uso de memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int i;
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads

  int locID = threadIdx.x;
  int xCent = w / 2;
  int yCent = h / 2;

  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  __shared__ int localAcc[degreeBins * rBins];

  for (i = locID; i < degreeBins * rBins; i += blockDim.x)
    localAcc[i] = 0;
  
  // esperar a que todos los threads hagan la inicializacion
  __syncthreads();

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx ++)
        {
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          atomicAdd (localAcc + (rIdx * degreeBins + tIdx), 1);
        }
    }
  
    // esepramos a que todos los warps terminen los calculos
    __syncthreads();

    // atomic add para acumular lo local en lo global
    for (i = locID; i < degreeBins * rBins; i+= blockDim.x)
      atomicAdd (acc + i, localAcc[i]);
}

//TODO Kernel memoria Constante
__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx]; //probar con esto para ver diferencia en tiempo
          int rIdx = (r + rMax) / rScale;
          // Debemos usar atomic, debido a que en los pixeles
          // más cercanos al punto central se puede generar
          // race conditions debido al concepto de línea
          // en cuanto a gráficas por computador se refiere.
          // ya que estos son valores discretos y no continuos
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  //TODO calcular: int gloID = ?
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  //Variables utilizadas para medir el tiempo de ejecucion con cuda events
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global

  // CONSTANT MEMORIES CHANES
  //
  //
  //
  cudaMemcpyToSymbol (d_Cos, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol (d_Sin, pcSin, sizeof (float) * degreeBins);
  //
  //
  //
  //

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);

  cudaEventRecord(start);
  GPU_HoughTranShared <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  cudaEventRecord(stop);
  cudaThreadSynchronize (); 
  
  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
    printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");
  
  printf("Execution time: %f milliseconds\n", ms);
  
  // TODO clean-up
  cudaFree ((void *) d_Cos);
  cudaFree ((void *) d_Sin);
  cudaFree ((void *) d_in);
  cudaFree ((void *) d_hough);
  free (h_hough);
  free (cpuht);
  free (pcCos);
  free (pcSin);
  cudaDeviceReset ();

  return 0;
}
