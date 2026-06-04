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