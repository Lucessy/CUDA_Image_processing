#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

#define BLOCK_SIZE_REDUCC 512  // Tamaño máximo de bloque para reducción

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;

// Estructura para representar un píxel
typedef struct  {
    byte b, g, r;
} Pixel;

// Estructura para representar los umbrales de un color específico
typedef struct {
    int r_min, r_max;
    int g_min, g_max;
    int b_min, b_max;
} ColorRange;

// Rangos de color en memoria constante
__constant__ ColorRange c_redRange;
__constant__ ColorRange c_greenRange;
__constant__ ColorRange c_blueRange;

// Valores mejorados para percepción humana
const ColorRange DEFAULT_RED = { 100, 255, 0, 150, 0, 150 };
const ColorRange DEFAULT_GREEN = { 30, 150, 50, 255, 0, 75 };
const ColorRange DEFAULT_BLUE = { 0, 200, 0, 250, 100, 255 };

// Rango para el color a filtrar
__constant__ ColorRange c_targetColorRange;  

// Función para leer una imagen BMP
void ReadImage(const char* fileName, Pixel** pixels, int32* width, int32* height, int32* bytesPerPixel) {
    FILE* imageFile = fopen(fileName, "rb");
    int32 dataOffset;
    fseek(imageFile, DATA_OFFSET_OFFSET, SEEK_SET);
    fread(&dataOffset, 4, 1, imageFile);
    fseek(imageFile, WIDTH_OFFSET, SEEK_SET);
    fread(width, 4, 1, imageFile);
    fseek(imageFile, HEIGHT_OFFSET, SEEK_SET);
    fread(height, 4, 1, imageFile);
    int16 bitsPerPixel;
    fseek(imageFile, BITS_PER_PIXEL_OFFSET, SEEK_SET);
    fread(&bitsPerPixel, 2, 1, imageFile);
    *bytesPerPixel = ((int32)bitsPerPixel) / 8;

    int paddedRowSize = (int)(4 * ceil((float)(*width) / 4.0f)) * (*bytesPerPixel);
    int unpaddedRowSize = (*width) * (*bytesPerPixel);
    int totalSize = unpaddedRowSize * (*height);
    *pixels = (Pixel*)malloc(totalSize);

    byte* currentRowPointer = (byte*)(*pixels) + ((*height - 1) * unpaddedRowSize);
    for (int i = 0; i < *height; i++) {
        fseek(imageFile, dataOffset + (i * paddedRowSize), SEEK_SET);
        fread(currentRowPointer, 1, unpaddedRowSize, imageFile);
        currentRowPointer -= unpaddedRowSize;
    }

    // En ReadImage, después de leer los píxeles:
    std::cout << "Primer píxel: R=" << (int)(*pixels)[0].r
        << " G=" << (int)(*pixels)[0].g
        << " B=" << (int)(*pixels)[0].b << "\n";

    fclose(imageFile);
}

// Función para escribir una imagen BMP
void WriteImage(const char* fileName, Pixel* pixels, int32 width, int32 height, int32 bytesPerPixel) {
    FILE* outputFile = fopen(fileName, "wb");
    const char* BM = "BM";
    fwrite(&BM[0], 1, 1, outputFile);
    fwrite(&BM[1], 1, 1, outputFile);

    int paddedRowSize = (int)(4 * ceil((float)width / 4.0f)) * bytesPerPixel;
    int32 fileSize = paddedRowSize * height + HEADER_SIZE + INFO_HEADER_SIZE;
    fwrite(&fileSize, 4, 1, outputFile);

    int32 reserved = 0x0000;
    fwrite(&reserved, 4, 1, outputFile);

    int32 dataOffset = HEADER_SIZE + INFO_HEADER_SIZE;
    fwrite(&dataOffset, 4, 1, outputFile);

    int32 infoHeaderSize = INFO_HEADER_SIZE;
    fwrite(&infoHeaderSize, 4, 1, outputFile);
    fwrite(&width, 4, 1, outputFile);
    fwrite(&height, 4, 1, outputFile);

    int16 planes = 1;
    fwrite(&planes, 2, 1, outputFile);

    int16 bitsPerPixel = bytesPerPixel * 8;
    fwrite(&bitsPerPixel, 2, 1, outputFile);

    int32 compression = NO_COMPRESION;
    fwrite(&compression, 4, 1, outputFile);

    int32 imageSize = width * height * bytesPerPixel;
    fwrite(&imageSize, 4, 1, outputFile);

    int32 resolutionX = 11811;
    int32 resolutionY = 11811;
    fwrite(&resolutionX, 4, 1, outputFile);
    fwrite(&resolutionY, 4, 1, outputFile);

    int32 colorsUsed = MAX_NUMBER_OF_COLORS;
    fwrite(&colorsUsed, 4, 1, outputFile);

    int32 importantColors = ALL_COLORS_REQUIRED;
    fwrite(&importantColors, 4, 1, outputFile);

    int unpaddedRowSize = width * bytesPerPixel;
    for (int i = 0; i < height; i++) {
        int pixelOffset = ((height - i) - 1) * unpaddedRowSize;
        fwrite(&((byte*)pixels)[pixelOffset], 1, paddedRowSize, outputFile);
    }

    fclose(outputFile);
}

void PrintPixelMatrix(Pixel* pixels, int width, int height, int maxRows = 10, int maxCols = 10) {
    std::cout << "Matriz de píxeles (mostrando los primeros " << maxRows << "x" << maxCols << " píxeles):" << std::endl;
    for (int y = 0; y < maxRows && y < height; y++) {
        for (int x = 0; x < maxCols && x < width; x++) {
            int idx = y * width + x;
            Pixel p = pixels[idx];
            std::cout << "(" << (int)p.r << "," << (int)p.g << "," << (int)p.b << ") ";
        }
        std::cout << std::endl;
    }
}

// Kernel CUDA para convertir una imagen a escala de grises
__global__ void GrayscaleKernel(Pixel* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x; // Índice del píxel en la matriz
        Pixel p = pixels[idx];   // Obtener el píxel en la posición (x, y)

        // Obtener los componentes RGB
        byte r = p.r;
        byte g = p.g;
        byte b = p.b;

        // Convertir a escala de grises
        byte gray = (byte)(0.299f * r + 0.587f * g + 0.114f * b);

        // Actualizar el píxel con el valor en escala de grises
        pixels[idx].r = gray;
        pixels[idx].g = gray;
        pixels[idx].b = gray;
    }
}

// Kernel CUDA para aplicar el filtro de pixelado
__global__ void PixelateKernel(Pixel* pixels, int width, int height, int filterSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int halfFilter = filterSize / 2;

        int rSum = 0, gSum = 0, bSum = 0;
        int count = 0;

        for (int dy = -halfFilter; dy <= halfFilter; dy++) {
            for (int dx = -halfFilter; dx <= halfFilter; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nIdx = ny * width + nx;
                    rSum += pixels[nIdx].r;
                    gSum += pixels[nIdx].g;
                    bSum += pixels[nIdx].b;
                    count++;
                }
            }
        }

        if (count > 0) {
            pixels[idx].r = rSum / count;
            pixels[idx].g = gSum / count;
            pixels[idx].b = bSum / count;
        }
    }
}

// Kernel para clasificación y conteo atómico
__global__ void ClassifyAndFilterColors(Pixel* pixels, int width, int height, int* redCount, int* greenCount, int* blueCount, Pixel* redOutput, Pixel* greenOutput, Pixel* blueOutput) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        Pixel p = pixels[idx];

        // Inicializar salidas con gris
        redOutput[idx] = { 255, 255, 255 };
        greenOutput[idx] = { 255, 255, 255 };
        blueOutput[idx] = { 255, 255, 255 };

        // Verificación de colores con operaciones atómicas
        bool isRed = (p.r >= c_redRange.r_min && p.r <= c_redRange.r_max &&
            p.g >= c_redRange.g_min && p.g <= c_redRange.g_max &&
            p.b >= c_redRange.b_min && p.b <= c_redRange.b_max);

        bool isGreen = (p.r >= c_greenRange.r_min && p.r <= c_greenRange.r_max &&
            p.g >= c_greenRange.g_min && p.g <= c_greenRange.g_max &&
            p.b >= c_greenRange.b_min && p.b <= c_greenRange.b_max);

        bool isBlue = (p.r >= c_blueRange.r_min && p.r <= c_blueRange.r_max &&
            p.g >= c_blueRange.g_min && p.g <= c_blueRange.g_max &&
            p.b >= c_blueRange.b_min && p.b <= c_blueRange.b_max);

        if (isRed) {
            atomicAdd(redCount, 1);
            redOutput[idx] = p;  // Conservar color original
        }
        else if (isGreen) {
            atomicAdd(greenCount, 1);
            greenOutput[idx] = p;   // Conservar color original
        }
        else if (isBlue) {
            atomicAdd(blueCount, 1);
            blueOutput[idx] = p;    // Conservar color original
        }
    }
}

// Kernel corregido para Filtrar y delinear un color
__global__ void FilterAndOutlineKernel(Pixel* input, Pixel* output, int width, int height, int* colorCount, int haloSize, bool useRed, bool useGreen, bool useBlue) {
    extern __shared__ Pixel sharedPixels[];

    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    // Dimensiones del bloque en memoria compartida (incluyendo halo)
    int sharedWidth = blockDim.x + 2 * haloSize;
    int sharedHeight = blockDim.y + 2 * haloSize;

    // Cargar datos en memoria compartida (todos los hilos participan)
    for (int dy = -haloSize; dy < blockDim.y + haloSize; dy += blockDim.y) {
        for (int dx = -haloSize; dx < blockDim.x + haloSize; dx += blockDim.x) {
            int loadX = globalX + dx;
            int loadY = globalY + dy;

            if (loadX >= 0 && loadX < width && loadY >= 0 && loadY < height) {
                int sharedIdx = (localY + dy + haloSize) * sharedWidth + (localX + dx + haloSize);
                sharedPixels[sharedIdx] = input[loadY * width + loadX];
            }
        }
    }
    __syncthreads();

    if (globalX < width && globalY < height) {
        // Índice en memoria compartida (centro del bloque)
        int sharedIdx = (localY + haloSize) * sharedWidth + (localX + haloSize);
        Pixel p = sharedPixels[sharedIdx];

        // Determinar si es el color objetivo
        bool isTargetColor = false;
        if (useRed) {
            isTargetColor = (p.r >= c_targetColorRange.r_min && p.r <= c_targetColorRange.r_max &&
                p.g >= c_targetColorRange.g_min && p.g <= c_targetColorRange.g_max &&
                p.b >= c_targetColorRange.b_min && p.b <= c_targetColorRange.b_max);
        }
        else if (useGreen) {
            isTargetColor = (p.r >= c_targetColorRange.r_min && p.r <= c_targetColorRange.r_max &&
                p.g >= c_targetColorRange.g_min && p.g <= c_targetColorRange.g_max &&
                p.b >= c_targetColorRange.b_min && p.b <= c_targetColorRange.b_max);
        }
        else if (useBlue) {
            isTargetColor = (p.r >= c_targetColorRange.r_min && p.r <= c_targetColorRange.r_max &&
                p.g >= c_targetColorRange.g_min && p.g <= c_targetColorRange.g_max &&
                p.b >= c_targetColorRange.b_min && p.b <= c_targetColorRange.b_max);
        }

        // Convertir todo a blanco y negro primero
        byte gray = (byte)(0.299f * p.r + 0.587f * p.g + 0.114f * p.b);
        output[globalY * width + globalX] = { gray, gray, gray };

        // Si es color objetivo, mantener el color original y aplicar halo
        if (isTargetColor) {
            atomicAdd(colorCount, 1);
            output[globalY * width + globalX] = p;

            // Marcar píxeles de halo
            for (int dy = -haloSize; dy <= haloSize; dy++) {
                for (int dx = -haloSize; dx <= haloSize; dx++) {
                    if (dx != 0 || dy != 0) { // No modificar el píxel central
                        int haloX = globalX + dx;
                        int haloY = globalY + dy;

                        if (haloX >= 0 && haloX < width && haloY >= 0 && haloY < height) {
                            // Solo marcar como negro si no es otro píxel de color objetivo
                            Pixel haloPixel = input[haloY * width + haloX];
                            bool isHaloPixelTarget = false;

                            if (useRed) {
                                isHaloPixelTarget = (haloPixel.r >= c_targetColorRange.r_min && haloPixel.r <= c_targetColorRange.r_max &&
                                    haloPixel.g >= c_targetColorRange.g_min && haloPixel.g <= c_targetColorRange.g_max &&
                                    haloPixel.b >= c_targetColorRange.b_min && haloPixel.b <= c_targetColorRange.b_max);
                            }
                            else if (useGreen) {
                                isHaloPixelTarget = (haloPixel.r >= c_targetColorRange.r_min && haloPixel.r <= c_targetColorRange.r_max &&
                                    haloPixel.g >= c_targetColorRange.g_min && haloPixel.g <= c_targetColorRange.g_max &&
                                    haloPixel.b >= c_targetColorRange.b_min && haloPixel.b <= c_targetColorRange.b_max);
                            }
                            else if (useBlue) {
                                isHaloPixelTarget = (haloPixel.r >= c_targetColorRange.r_min && haloPixel.r <= c_targetColorRange.r_max &&
                                    haloPixel.g >= c_targetColorRange.g_min && haloPixel.g <= c_targetColorRange.g_max &&
                                    haloPixel.b >= c_targetColorRange.b_min && haloPixel.b <= c_targetColorRange.b_max);
                            }

                            if (!isHaloPixelTarget) {
                                output[haloY * width + haloX] = { 0, 0, 0 }; // Negro
                            }
                        }
                    }
                }
            }
        }
    }
}

// Función para encontrar el máximo entre dos valores
__device__ int max_int(int a, int b) {
    return (a > b) ? a : b;
}

// Kernel de reducción optimizado (versión 6 del documento NVIDIA)
template <unsigned int blockSize>
__global__ void reduce_max(int* g_idata, int* g_odata, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    int myMax = 0;

    // Cargar y reducir múltiples elementos por hilo
    while (i < n) {
        myMax = max_int(myMax, g_idata[i]);
        if (i + blockSize < n) {
            myMax = max_int(myMax, g_idata[i + blockSize]);
        }
        i += gridSize;
    }

    sdata[tid] = myMax;
    __syncthreads();

    // Reducción en memoria compartida
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] = max_int(sdata[tid], sdata[tid + 256]); }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] = max_int(sdata[tid], sdata[tid + 128]); }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] = max_int(sdata[tid], sdata[tid + 64]); }
        __syncthreads();
    }

    // Reducción final en el warp
    if (tid < 32) {
        volatile int* vsmem = sdata;
        if (blockSize >= 64) vsmem[tid] = max_int(vsmem[tid], vsmem[tid + 32]);
        if (blockSize >= 32) vsmem[tid] = max_int(vsmem[tid], vsmem[tid + 16]);
        if (blockSize >= 16) vsmem[tid] = max_int(vsmem[tid], vsmem[tid + 8]);
        if (blockSize >= 8) vsmem[tid] = max_int(vsmem[tid], vsmem[tid + 4]);
        if (blockSize >= 4) vsmem[tid] = max_int(vsmem[tid], vsmem[tid + 2]);
        if (blockSize >= 2) vsmem[tid] = max_int(vsmem[tid], vsmem[tid + 1]);
    }

    // Escribir resultado para este bloque
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Función para normalizar valores a rango ASCII (35-125)
int normalizeToASCII(int value) {
    return 35 + (int)((float)value / 255.0f * 90.0f);
}

// Kernel para transformar píxeles a valores numéricos
__global__ void transformPixels(Pixel* pixels, int* values, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width * height) {
        Pixel p = pixels[idx];
        values[idx] = (int)(p.r * 0.5f + p.g * 0.25f + p.b * 0.25f);
    }
}

// Implementación optimizada de la opción 5 (pseudo-hash)
void computeImageHash(Pixel* h_pixels, int width, int height) {
    int totalPixels = width * height;

    // Encuentra el valor máximo real en la imagen (en CPU)
    int true_max = 0;
    for (int i = 0; i < width * height; i++) {
        int val = (int)(0.5 * h_pixels[i].r + 0.25 * h_pixels[i].g + 0.25 * h_pixels[i].b);
        if (val > true_max) true_max = val;
    }
    std::cout << "Máximo real esperado: " << true_max << "\n";

    // 1. Reservar memoria en device
    Pixel* d_image;
    int* d_values, * d_partialSums;

    cudaMalloc(&d_image, totalPixels * sizeof(Pixel));
    cudaMalloc(&d_values, totalPixels * sizeof(int));
    cudaMalloc(&d_partialSums, totalPixels * sizeof(int));

    // Copiar imagen a device
    cudaMemcpy(d_image, h_pixels, totalPixels * sizeof(Pixel), cudaMemcpyHostToDevice);

    // 2. Transformar imagen a valores numéricos
    dim3 blockDim(512);  // Tamaño de bloque óptimo
    dim3 gridDim((totalPixels + blockDim.x - 1) / blockDim.x);
    // Kernel para transformar píxeles a valores
    transformPixels << <gridDim, blockDim >> > (d_image, d_values, width, height);
    cudaDeviceSynchronize();

    // Verificación después de transformPixels
    int* h_temp = (int*)malloc(totalPixels * sizeof(int));
    cudaMemcpy(h_temp, d_values, totalPixels * sizeof(int), cudaMemcpyDeviceToHost);

    // Verificar primeros 10 valores
    std::cout << "Primeros 10 valores transformados:\n";
    for (int i = 0; i < 10 && i < totalPixels; i++) {
        Pixel p = h_pixels[i];
        int expected = (int)(0.5f * p.r + 0.25f * p.g + 0.25f * p.b);
        std::cout << "Pixel " << i << ": (" << (int)p.r << "," << (int)p.g << "," << (int)p.b
            << ") -> " << h_temp[i] << " (esperado: " << expected << ")\n";
    }
    free(h_temp);

    // 3. Reducción paralela (múltiples pasos)
    int* currentInput = d_values;
    int* currentOutput = d_partialSums;
    int currentSize = totalPixels;
    int iterations = 0;

    std::cout << "\nIniciando reducción...\n";
    std::cout << "Ancho: " << width << ", Alto: " << height << "\n";
    std::cout << "Total píxeles: " << totalPixels << "\n";
    int blockSize = 512;
    int numBlocks = currentSize / blockSize;
    std::cout << "Reducimos con " << numBlocks << " bloques\n";
    reduce_max<512><<<numBlocks, blockSize>>>(currentInput, currentOutput, currentSize);
    cudaDeviceSynchronize();
    std::swap(currentInput, currentOutput);
    currentSize = numBlocks;
    std::cout << "---> Tenemos: " << currentSize << " valores\n";

    blockSize = 15;
    while (currentSize > 15) {
        numBlocks = (currentSize + blockSize - 1) / blockSize;
        std::cout << "Reducimos con " << numBlocks << " bloques\n";

        // Lanzar kernel de reducción con el tamaño de bloque apropiado
        reduce_max<15> << <numBlocks, blockSize>> > (currentInput, currentOutput, currentSize);
        cudaDeviceSynchronize();

        // Intercambiar punteros para la siguiente iteración
        std::swap(currentInput, currentOutput);
        currentSize = numBlocks;

        std::cout << "---> Tenemos: " << currentSize << " valores\n";
    }

    // 4. Copiar resultados finales
    int finalValues[512];
    cudaMemcpy(finalValues, currentInput, currentSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Normalizar a ASCII y mostrar resultados
    std::cout << "\nHash [ASCII]: ";
    for (int i = 0; i < 10; i++) {
        std::cout << (char)normalizeToASCII(finalValues[i]) << " ";
    }

    std::cout << "\nHash [núm. normalizado]: ";
    for (int i = 0; i < 10; i++) {
        std::cout << normalizeToASCII(finalValues[i]) << " ";
    }

    std::cout << "\nHash [núm. original]: ";
    for (int i = 0; i < 10; i++) {
        std::cout << finalValues[i] << " ";
    }
    std::cout << "\n";

    // 6. Liberar memoria
    cudaFree(d_image);
    cudaFree(d_values);
    cudaFree(d_partialSums);
}



int main() {
    int option;
    std::cout << "Seleccione una opción:\n";
    std::cout << "1. Aplicar filtro en blanco y negro\n";
    std::cout << "2. Aplicar filtro pixelado en color\n";
    std::cout << "3. Aplicar filtro de colores\n";
    std::cout << "4. Aplicar halo a foto blanco y negro con un filtro de color\n";
    std::cout << "5. Obtener pseudo-hash de la imagen\n";
    std::cin >> option;

    Pixel* h_pixels, * d_pixels;
    int32 width, height, bytesPerPixel;

    // Leer la imagen BMP
    ReadImage("input.bmp", &h_pixels, &width, &height, &bytesPerPixel);

    // Imprimir la matriz de píxeles
    //PrintPixelMatrix(h_pixels, width, height,5,5);

    if (option == 1) {

        // Asignar memoria en la GPU
        cudaMalloc((void**)&d_pixels, width * height * sizeof(Pixel));
        cudaMemcpy(d_pixels, h_pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        // Configurar el kernel
        dim3 blockSize(1024); // Bloques de 16x16 hilos (256 hilos por bloque)
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // Llamar al kernel para convertir a escala de grises
        GrayscaleKernel << <gridSize, blockSize >> > (d_pixels, width, height);

        // Copiar los datos de vuelta a la CPU
        cudaMemcpy(h_pixels, d_pixels, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);
        // Guardar la imagen resultante
        WriteImage("outputBlackAndWhite.bmp", h_pixels, width, height, bytesPerPixel);
        // Liberar memoria
        cudaFree(d_pixels);
        free(h_pixels);
        printf("Proceso completado. Imagen guardada como 'outputBlackAndWhite.bmp'.\n");
    }
    else if (option == 2) {
        // Asignar memoria en la GPU
        cudaMalloc((void**)&d_pixels, width * height * sizeof(Pixel));
        cudaMemcpy(d_pixels, h_pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        // Configurar el kernel
        dim3 blockSize(1024); // Bloques de 16x16 hilos (256 hilos por bloque)
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        int filterSize = 30;
        // Llamar al kernel para convertir a escala de grises
        PixelateKernel << <gridSize, blockSize >> > (d_pixels, width, height, filterSize);

        // Copiar los datos de vuelta a la CPU
        cudaMemcpy(h_pixels, d_pixels, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);
        // Guardar la imagen resultante
        WriteImage("outputPixelate.bmp", h_pixels, width, height, bytesPerPixel);
        // Liberar memoria
        cudaFree(d_pixels);
        free(h_pixels);
        printf("Proceso completado. Imagen guardada como 'outputPixelate.bmp'.\n");
    }
    else if (option == 3) {
        // Configuración de rangos (usando valores por defecto)
        ColorRange redRange = DEFAULT_RED;
        ColorRange greenRange = DEFAULT_GREEN;
        ColorRange blueRange = DEFAULT_BLUE;

        std::cout << "Ingrese umbrales para rojo (r_min r_max g_min g_max b_min b_max): ";
        //std::cin >> redRange.r_min >> redRange.r_max >> redRange.g_min >> redRange.g_max >> redRange.b_min >> redRange.b_max;

        std::cout << "Ingrese umbrales para verde (r_min r_max g_min g_max b_min b_max): ";
        //std::cin >> greenRange.r_min >> greenRange.r_max >> greenRange.g_min >> greenRange.g_max >> greenRange.b_min >> greenRange.b_max;

        std::cout << "Ingrese umbrales para azul (r_min r_max g_min g_max b_min b_max): ";
        //std::cin >> blueRange.r_min >> blueRange.r_max >> blueRange.g_min >> blueRange.g_max >> blueRange.b_min >> blueRange.b_max;

        // Copiar rangos a memoria constante
        cudaMemcpyToSymbol((void*)&c_redRange, &redRange, sizeof(ColorRange));
        cudaMemcpyToSymbol((void*)&c_greenRange, &greenRange, sizeof(ColorRange));
        cudaMemcpyToSymbol((void*)&c_blueRange, &blueRange, sizeof(ColorRange));

        // Variables para conteo
        int* d_redCount, * d_greenCount, * d_blueCount;
        int h_redCount = 0, h_greenCount = 0, h_blueCount = 0;

        cudaMalloc(&d_redCount, sizeof(int));
        cudaMalloc(&d_greenCount, sizeof(int));
        cudaMalloc(&d_blueCount, sizeof(int));
        cudaMemset(d_redCount, 0, sizeof(int));
        cudaMemset(d_greenCount, 0, sizeof(int));
        cudaMemset(d_blueCount, 0, sizeof(int));

        // Buffers para imágenes de salida
        Pixel* d_pixels, * d_redOutput, * d_greenOutput, * d_blueOutput;
        cudaMalloc(&d_pixels, width * height * sizeof(Pixel));
        cudaMalloc(&d_redOutput, width * height * sizeof(Pixel));
        cudaMalloc(&d_greenOutput, width * height * sizeof(Pixel));
        cudaMalloc(&d_blueOutput, width * height * sizeof(Pixel));

        // Copiar imagen original a GPU
        cudaMemcpy(d_pixels, h_pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        // Ejecutar kernel
        dim3 blockSize(1024);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        ClassifyAndFilterColors << <gridSize, blockSize >> > (d_pixels, width, height, d_redCount, d_greenCount, d_blueCount, d_redOutput, d_greenOutput, d_blueOutput);

        // Copiar resultados
        cudaMemcpy(&h_redCount, d_redCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_greenCount, d_greenCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_blueCount, d_blueCount, sizeof(int), cudaMemcpyDeviceToHost);

        Pixel* h_redOutput = (Pixel*)malloc(width * height * sizeof(Pixel));
        Pixel* h_greenOutput = (Pixel*)malloc(width * height * sizeof(Pixel));
        Pixel* h_blueOutput = (Pixel*)malloc(width * height * sizeof(Pixel));

        cudaMemcpy(h_redOutput, d_redOutput, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_greenOutput, d_greenOutput, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_blueOutput, d_blueOutput, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

        // Mostrar resultados
        std::cout << "\nUmbrales usados:\n";
        std::cout << "Rojo: R(" << redRange.r_min << "-" << redRange.r_max << ") "
            << "G(" << redRange.g_min << "-" << redRange.g_max << ") "
            << "B(" << redRange.b_min << "-" << redRange.b_max << ")\n";
        std::cout << "Verde: R(" << greenRange.r_min << "-" << greenRange.r_max << ") "
            << "G(" << greenRange.g_min << "-" << greenRange.g_max << ") "
            << "B(" << greenRange.b_min << "-" << greenRange.b_max << ")\n";
        std::cout << "Azul: R(" << blueRange.r_min << "-" << blueRange.r_max << ") "
            << "G(" << blueRange.g_min << "-" << blueRange.g_max << ") "
            << "B(" << blueRange.b_min << "-" << blueRange.b_max << ")\n\n";

        std::cout << "Píxeles detectados:\n";
        std::cout << "Rojo: " << h_redCount << " píxeles\n";
        std::cout << "Verde: " << h_greenCount << " píxeles\n";
        std::cout << "Azul: " << h_blueCount << " píxeles\n";

        // Guardar imágenes
        WriteImage("red_filtered.bmp", h_redOutput, width, height, bytesPerPixel);
        WriteImage("green_filtered.bmp", h_greenOutput, width, height, bytesPerPixel);
        WriteImage("blue_filtered.bmp", h_blueOutput, width, height, bytesPerPixel);

        // Liberar memoria
        cudaFree(d_pixels);
        cudaFree(d_redOutput);
        cudaFree(d_greenOutput);
        cudaFree(d_blueOutput);
        cudaFree(d_redCount);
        cudaFree(d_greenCount);
        cudaFree(d_blueCount);
        free(h_redOutput);
        free(h_greenOutput);
        free(h_blueOutput);

        printf("Proceso completado. Imagen guardada como 'red_filtered.bmp'.\n");
        printf("Proceso completado. Imagen guardada como 'green_filtered.bmp'.\n");
        printf("Proceso completado. Imagen guardada como 'blue_filtered.bmp'.\n");
    }
    else if (option == 4) {
        // Obtener propiedades del dispositivo
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int sharedMemPerBlock = prop.sharedMemPerBlock;

        // Interacción con el usuario
        int colorChoice, haloSize;
        std::cout << "Seleccione color a filtrar (1=Rojo, 2=Verde, 3=Azul): ";
        std::cin >> colorChoice;
        std::cout << "Ingrese tamaño del halo (1-5 pixeles recomendado): ";
        std::cin >> haloSize;

        // Configurar rango de color según selección
        ColorRange targetRange;
        switch (colorChoice) {
        case 1: targetRange = DEFAULT_RED; break;
        case 2: targetRange = DEFAULT_GREEN; break;
        case 3: targetRange = DEFAULT_BLUE; break;
        default: targetRange = DEFAULT_RED; break;
        }
        cudaMemcpyToSymbol(c_targetColorRange, &targetRange, sizeof(ColorRange));

        // Asignar memoria en la GPU
        Pixel* d_input, * d_output;
        cudaMalloc(&d_input, width * height * sizeof(Pixel));
        cudaMalloc(&d_output, width * height * sizeof(Pixel));
        cudaMemcpy(d_input, h_pixels, width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

        // Calcular dimensiones óptimas
        int blockSizeX = 32;
        int blockSizeY = 32;
        int sharedMemRequired = (blockSizeX + 2 * haloSize) * (blockSizeY + 2 * haloSize) * sizeof(Pixel);

        // Ajustar tamaño de bloque si se necesita demasiada memoria compartida
        while (sharedMemRequired > sharedMemPerBlock && blockSizeX > 8) {
            blockSizeX /= 2;
            blockSizeY /= 2;
            sharedMemRequired = (blockSizeX + 2 * haloSize) * (blockSizeY + 2 * haloSize) * sizeof(Pixel);
        }

        dim3 blockSize(blockSizeX, blockSizeY);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);

        // Variables para conteo
        int* d_colorCount, h_colorCount = 0;
        cudaMalloc(&d_colorCount, sizeof(int));
        cudaMemset(d_colorCount, 0, sizeof(int));

        // Ejecutar kernel
        FilterAndOutlineKernel << <gridSize, blockSize, sharedMemRequired >> > (
            d_input, d_output, width, height, d_colorCount,
            haloSize, (colorChoice == 1), (colorChoice == 2), (colorChoice == 3));

        // Copiar resultado a CPU
        cudaMemcpy(h_pixels, d_output, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_colorCount, d_colorCount, sizeof(int), cudaMemcpyDeviceToHost);

        // Guardar imagen resultante
        WriteImage("output_with_halo.bmp", h_pixels, width, height, bytesPerPixel);

        // Liberar memoria
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_colorCount);

        std::cout << "Proceso completado. Imagen guardada como 'output_with_halo.bmp'.\n";
        std::cout << "Píxeles del color seleccionado: " << h_colorCount << std::endl;
    }
    else if (option == 5) {
        computeImageHash(h_pixels, width, height);
    }

    return 0;
}