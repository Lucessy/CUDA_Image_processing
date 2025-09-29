# Proyecto CUDA – Procesamiento Paralelo de Imágenes

## 📌 Descripción

Este proyecto implementa **cinco fases de manipulación de imágenes en formato BMP** utilizando **programación paralela con CUDA**.  
El objetivo principal es demostrar cómo la **GPU puede acelerar operaciones gráficas intensivas**, adaptándose dinámicamente a las capacidades de hardware.

Todo el desarrollo está centralizado en un único archivo: **`kernel.cu`**.

---

## ⚙️ Funcionalidades

El sistema permite ejecutar diferentes fases desde un **menú interactivo**, aplicando sobre una imagen de entrada distintas transformaciones:

1. **Conversión a Blanco y Negro**  
   - Transformación RGB → Escala de grises usando la fórmula perceptual:  
     `gray = 0.299 * R + 0.587 * G + 0.114 * B`  

2. **Filtro Pixelado Cristalizado**  
   - Efecto de pixelado configurable (`filterSize`).  
   - Uso de memoria compartida para optimización.  
   - Opción de aplicar en **color** o **escala de grises**.  

3. **Identificación de Colores (Rojo, Verde y Azul)**  
   - Umbrales por defecto y posibilidad de personalización.  
   - Uso de **memoria constante** para almacenar rangos RGB.  
   - Contadores atómicos para medir la cantidad de píxeles detectados.  

4. **Halo sobre Color Objetivo en Blanco y Negro**  
   - Destaca un color (rojo, verde o azul) manteniéndolo en color y aplicando un halo negro.  
   - El resto de la imagen se convierte en blanco y negro.  
   - Uso intensivo de **memoria compartida** y halo configurable (`haloSize`).  

5. **Reducción Paralela y Hash ASCII**  
   - Conversión RGB → valor escalar ponderado.  
   - Reducción jerárquica en GPU hasta obtener un conjunto reducido de valores.  
   - Generación de un **pseudo-hash ASCII** representando la imagen.  

---

## 📂 Estructura del Proyecto

```
├── kernel.cu   # Archivo principal con todas las fases CUDA
├── README.md   # Este archivo
```

---

## 🚀 Requisitos

- **CUDA Toolkit** (>= 11.0)  
- **NVIDIA GPU compatible** con Compute Capability 5.0 o superior  
- Compilador `nvcc`  

---

## ▶️ Ejecución

1. Compilar el proyecto:

```bash
nvcc -o procesamiento kernel.cu
```

2. Ejecutar el programa indicando la imagen de entrada:

```bash
./procesamiento imagen.bmp
```

3. Desde el menú interactivo, seleccionar la fase a ejecutar y configurar parámetros (filtros, umbrales, halo, etc.).

---

## 📊 Resultados
Los resultados obtenidos del código se encuentran en la memoria del proyecto.
- **Imágenes generadas:**  
  - `grayscale.bmp`  
  - `pixelated.bmp`  
  - `red_filtered.bmp`, `green_filtered.bmp`, `blue_filtered.bmp`  
  - `output_with_halo.bmp`  
- **Hash ASCII generado en terminal** para la fase 5.  

---

## ✨ Características Técnicas Destacadas

- Uso de **memoria compartida** para acelerar acceso a píxeles vecinos.  
- **Memoria constante** para parámetros configurables (umbrales RGB).  
- **Contadores atómicos** para estadísticas de clasificación de píxeles.  
- **Reducción jerárquica** optimizada en GPU.  
- Configuración **dinámica** de grid y block size según hardware.  

---

## 👩‍💻 Autora

- **Luciana Paola Díaz**  
  Universidad de Alcalá – Escuela Politécnica Superior  
  Paradigmas Avanzados de Programación – 2025  
