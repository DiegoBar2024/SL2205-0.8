# Seretest

Este proyecto, **Seretest**, está enfocado en el análisis y estimación de parámetros de la marcha, detección de actividades y la implementación de modelos de inteligencia artificial aplicados al análisis de señales.

---

## Estructura de la Carpeta `sereTestLib`

A continuación se muestra la estructura de la carpeta **`sereTestLib`** con sus principales subcarpetas:

<!-- TREEVIEW START -->
```bash
├── sereTestLib/
│   ├── Cinematica/
│   ├── Largo Plazo/
│   ├── Presentacion/
│   ├── Principal/
│   ├── Redes Neuronales/
│   ├── Wavelets/
│   └── parameters.py
```
<!-- TREEVIEW END -->

## Explicación de la Carpeta `sereTestLib`

La carpeta **`sereTestLib`** es el núcleo de la lógica y funcionalidades principales del proyecto **Seretest**. Contiene los módulos y scripts que implementan las diversas capacidades del proyecto, como el análisis de la marcha, la detección de actividades, el entrenamiento de modelos de inteligencia artificial y el procesamiento de señales. A continuación se describe brevemente el contenido de cada subcarpeta y archivo en **`sereTestLib`**:

### **`Cinematica`**
Contiene las herramientas para la estimación de **parámetros de marcha** como la longitud de paso, duración de paso, velocidad y cadencia. También se encarga de la **detección de eventos de marcha**, tales como los contactos iniciales (heel strikes), contactos terminales (toe offs) y giros durante la marcha.

### **`Largo Plazo`**
Se enfoca en la **detección de actividades**, como la clasificación entre **reposo** y **movimiento**, además de realizar la **detección de anomalías** tanto en la marcha como en las actividades asociadas.

### **`Presentacion`**
Contiene archivos y módulos para la **generación de reportes visuales y gráficos**. Su propósito es facilitar la exportación de los resultados obtenidos en el análisis en formatos como **PDF**.

### **`Principal`**
Contiene los **módulos de interfaz** que permiten a los usuarios interactuar con el programa de manera más sencilla, proporcionando una forma amigable de acceder a las funcionalidades del proyecto.

### **`Redes Neuronales`**
En esta carpeta se encuentran los archivos relacionados con el **entrenamiento y validación de modelos de inteligencia artificial**, incluyendo **autoencoders** para compresión de datos, y técnicas de clasificación como **LSTM** (Long Short-Term Memory) y **DTW** (Dynamic Time Warping) aplicadas en clasificación de imágenes y secuencias de imágenes.

### **`Wavelets`**
Aquí se encuentran los archivos dedicados al procesamiento de señales mediante la **segmentación en ventanas**, el cálculo de **escalogramas** para cada ventana y el análisis de la **energía** de los escalogramas, permitiendo un análisis detallado de las señales a diferentes escalas.

### **`parameters.py`**
Archivo de configuración global que contiene los **parámetros** utilizados en todo el proyecto, como valores constantes, hiperparámetros para los modelos de inteligencia artificial y configuraciones generales para el ajuste del comportamiento del proyecto.