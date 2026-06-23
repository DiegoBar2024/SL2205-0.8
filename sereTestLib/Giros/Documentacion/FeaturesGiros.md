# Features extraídas de los segmentos de giro unidimensionales

Esta sección describe las características extraídas de un segmento de señal unidimensional $x = \{x_1, ..., x_N\}$ correspondiente a un eje del giroscopio o acelerómetro dentro de un evento de giro.

Cada feature captura propiedades estadísticas, dinámicas o espectrales del movimiento.

## 1. Media

$$
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

Representa el valor promedio del segmento de señal. Captura el nivel global de la señal durante el movimiento.

## 2. Pico absoluto (Peak)

$$
\text{peak} = \max_i |x_i|
$$

Mide la amplitud máxima absoluta alcanzada por la señal. Refleja la intensidad máxima del movimiento en el segmento.

## 3. Energía RMS (Root Mean Square)

$$
\text{rms} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}
$$

Representa la energía efectiva de la señal. Es sensible tanto a la amplitud como a la duración del movimiento.

## 4. Tiempo al pico

$$
t_{peak} = \frac{\arg\max_i |x_i|}{N}
$$

Indica la posición relativa dentro del segmento en la que ocurre el máximo absoluto. Está normalizado entre 0 y 1.

## 5. Relación pico-media

$$
\text{peak\_mean\_ratio} = \frac{\text{peak}}{|\mu| + \varepsilon}
$$

Mide cuán dominante es el valor máximo respecto al nivel promedio de la señal. Valores altos indican movimientos más impulsivos o con picos aislados.

## 6. Asimetría (Skewness)

$$
\text{skew}(x) = \frac{\mathbb{E}\left[(x - \mu)^3\right]}{\sigma^3}
$$

La **skewness** (calculada mediante `scipy.stats.skew`) corresponde al **tercer momento central estandarizado** de la señal.

- Cuantifica la asimetría de la distribución de los valores dentro del segmento.
- Valores positivos indican una cola hacia valores altos.
- Valores negativos indican una cola hacia valores bajos.
- Valores cercanos a cero indican simetría.

Permite detectar si el movimiento está dominado por picos direccionales asimétricos.

## 7. Curtosis (Kurtosis)

$$
\text{kurtosis}(x) = \frac{\mathbb{E}\left[(x - \mu)^4\right]}{\sigma^4}
$$

En `scipy.stats.kurtosis`, por defecto se utiliza la definición de Fisher:

$$
\text{kurtosis}_{Fisher}(x) = \frac{\mathbb{E}\left[(x - \mu)^4\right]}{\sigma^4} - 3
$$

- Cuarto momento central estandarizado.
- Mide concentración de energía en colas y picos.
- Valores altos indican presencia de impulsos o outliers.
- Valores cercanos a 0 (Fisher) indican comportamiento similar a Gaussiana.

## 8. Tasa de cruces por cero (ZCR)

$$
\text{ZCR} = \frac{1}{N-1} \sum_{i=1}^{N-1} \mathbb{1}[\text{sign}(x_i) \neq \text{sign}(x_{i+1})]
$$

Cuantifica la frecuencia de cambios de signo en la señal. Se asocia con comportamiento oscilatorio o alternante.

## 9. Entropía espectral

Sea la transformada de Fourier:

$$
X_i = |\mathcal{F}(x)_i|
$$

Distribución normalizada:

$$
p_i = \frac{X_i}{\sum_j X_j + \varepsilon}
$$

Entropía:

$$
H = - \sum_i p_i \log(p_i + \varepsilon)
$$

Mide la dispersión de energía en el dominio de frecuencia. Valores altos indican señal más “ruidosa” o distribuida.

## 10. Energía de jerk

Derivada discreta:

$$
j_i = (x_{i+1} - x_i)\cdot f_s
$$

Energía:

$$
E_j = \frac{1}{N-1} \sum_{i=1}^{N-1} j_i^2
$$

Captura la intensidad de cambios rápidos en la señal. Valores altos indican movimientos más bruscos o menos suaves.

## 11. Energía de la señal

Para un segmento de señal $x = \{x_1, ..., x_N\}$, la energía total se define como:

$$
E = \sum_{i=1}^{N} x_i^2
$$

Esta métrica cuantifica la energía total acumulada en el segmento de señal.

- Valores altos indican movimientos de mayor amplitud sostenida.
- Valores bajos corresponden a movimientos más pequeños o menos intensos.
- A diferencia de la energía RMS, esta característica no está normalizada por la duración del segmento, por lo que combina información tanto de amplitud como de longitud temporal del movimiento.

En señales de velocidad angular y aceleración, la energía permite caracterizar la cantidad total de actividad mecánica presente durante un evento de giro.

## 12. Centroide espectral

El centroide espectral se define como el “centro de masa” del espectro de magnitud:

$$
C = \frac{\sum_i f_i X_i}{\sum_i X_i + \varepsilon}
$$

donde:
- $f_i$ es la frecuencia asociada al bin espectral $i$
- $X_i = |\mathcal{F}(x)_i|$ es la magnitud espectral

Esta métrica indica la localización promedio de la energía en el dominio de frecuencia.

- Valores bajos indican predominancia de bajas frecuencias (movimientos lentos o suaves).
- Valores altos indican presencia de altas frecuencias (movimientos rápidos o abruptos).
- En señales IMU, se interpreta como un indicador de “rapidez cinemática” del movimiento.

## 13. Dominancia espectral

La dominancia espectral cuantifica cuán concentrada está la energía en el espectro.

Primero se define la **planitud espectral**:

$$
F = \frac{\left(\prod_i X_i\right)^{1/N}}{\frac{1}{N}\sum_i X_i + \varepsilon}
$$

A partir de ella, la dominancia espectral se define como su inversa conceptual:

$$
D = \frac{1}{F + \varepsilon}
$$

Esta métrica mide el grado de concentración energética en pocas componentes frecuenciales.

- Valores altos indican espectros dominados por pocas frecuencias (señales estructuradas).
- Valores bajos indican espectros más planos (ruido o actividad distribuida).
- En giros IMU, ayuda a diferenciar movimientos organizados de patrones más complejos o inestables.

## 14. Características multirresolución basadas en wavelets (Relative Wavelet Energy - RWE)

Esta sección introduce un nuevo conjunto de características basadas en el análisis multirresolución mediante wavelets discretas, aplicado a señales unidimensionales $x = \{x_1, ..., x_N\}$ de acelerómetro o giroscopio.

El objetivo es capturar cómo se distribuye la energía de la señal a través de distintas escalas temporales (bandas de frecuencia implícitas), lo cual permite caracterizar estructuras dinámicas no estacionarias del movimiento.

### 14.1 Descomposición wavelet

Se utiliza la transformada wavelet discreta (DWT):

$$
\mathcal{W}(x) \rightarrow \{A_L, D_L, D_{L-1}, ..., D_1\}
$$

donde:
- $A_L$: coeficientes de aproximación (bajas frecuencias)
- $D_l$: coeficientes de detalle en el nivel $l$

### 14.2 Energía por subbanda

La energía de cada subbanda se define como:

$$
E_l = \sum_i D_l(i)^2
$$

La energía total de detalles es:

$$
E_{tot} = \sum_{l=1}^{L} E_l
$$

### 14.3 Relative Wavelet Energy (RWE)

La energía relativa de cada subbanda se define como:

$$
\text{RWE}_l = \frac{E_l}{E_{tot} + \varepsilon}
$$

Esta normalización permite comparar señales independientemente de su amplitud absoluta.

### 14.4 Agrupación fisiológica de bandas

Para interpretación cinemática se agrupan las subbandas en regiones funcionales:

- Alta frecuencia (HF): transitorios rápidos, cambios bruscos
- Media frecuencia (MF): dinámica intermedia del movimiento
- Baja frecuencia (LF): estructura global del giro

$$
\text{HF} = \text{RWE}_{D1} + \text{RWE}_{D2}
$$

$$
\text{MF} = \text{RWE}_{D3}
$$

$$
\text{LF} = \text{RWE}_{D4}
$$

### 14.5 Features derivadas

A partir de estas bandas se definen características compactas:

#### Energía relativa en alta frecuencia

$$
\text{rwe\_hf} = \text{HF}
$$

Captura la proporción de energía asociada a cambios rápidos y transitorios del movimiento.

#### Energía relativa en media frecuencia

$$
\text{rwe\_mf} = \text{MF}
$$

Representa la componente intermedia del movimiento, asociada a la dinámica principal del giro.

#### Energía relativa en baja frecuencia

$$
\text{rwe\_lf} = \text{LF}
$$

Captura la estructura lenta y global del movimiento.

#### Relación HF/LF

$$
\text{rwe\_hf\_lf\_ratio} = \frac{\text{HF}}{\text{LF} + \varepsilon}
$$

Mide el balance entre actividad rápida y estructura lenta del giro.

#### Balance espectral wavelet

$$
\text{rwe\_balance} = \text{HF} - \text{LF}
$$

Resume la dominancia relativa entre componentes rápidas y lentas.

### 14.6 Interpretación en análisis de movimiento

Estas características permiten caracterizar diferencias en patrones motores:

- Movimientos más bruscos o inestables tienden a incrementar HF
- Movimientos más suaves o controlados incrementan LF
- Diferencias en HF/LF pueden reflejar cambios en control motor asociados a edad o condición neuromotora

Estas métricas son complementarias a la energía de jerk, ya que:
- Jerk captura derivadas temporales locales (dominio temporal)
- RWE captura distribución de energía multiescala (dominio frecuencia-tiempo)

Por lo tanto, aportan información estructural parcialmente ortogonal al resto del conjunto de features.

## 15. Desviación estándar (Standard Deviation)

La desviación estándar mide la dispersión de los valores de la señal respecto a su media.

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}
$$

donde:
- $\mu$ es la media de la señal
- $x_i$ son los valores del segmento

### Interpretación

- Valores altos indican alta variabilidad en el movimiento
- Valores bajos indican señales más estables o constantes
- Es una medida global de dispersión alrededor del valor medio

En señales IMU (giroscopio y acelerómetro), la desviación estándar captura la intensidad de fluctuaciones del movimiento durante el giro.

## 16. Rango intercuartil (Interquartile Range - IQR)

El rango intercuartílico mide la dispersión central de la señal, ignorando valores extremos.

Se define como:

$$
IQR = Q_3 - Q_1
$$

donde:
- $Q_1$ es el primer cuartil (percentil 25)
- $Q_3$ es el tercer cuartil (percentil 75)

### Interpretación

- Es robusto frente a outliers
- Captura la variabilidad “típica” del movimiento
- Valores altos indican señales con amplia dispersión central
- Valores bajos indican señales más concentradas

En análisis de movimiento, el IQR es especialmente útil cuando existen picos o artefactos que pueden sesgar otras métricas como la desviación estándar.

## 17. Coeficiente de variación (Coefficient of Variation - CV)

El coeficiente de variación normaliza la dispersión respecto a la media de la señal.

Se define como:

$$
CV = \frac{\sigma}{|\mu| + \varepsilon}
$$

donde:
- $\sigma$ es la desviación estándar
- $\mu$ es la media de la señal

### Interpretación

- Permite comparar variabilidad entre señales con diferentes escalas
- Valores altos indican alta variabilidad relativa
- Valores bajos indican comportamiento más estable respecto al nivel medio

En señales de giroscopio y acelerómetro, el CV es especialmente útil para comparar sujetos o movimientos con amplitudes muy diferentes.