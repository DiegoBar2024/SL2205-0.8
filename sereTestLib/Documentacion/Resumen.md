# Resumen del Proyecto: Seretest

## Introducción y motivación

Seretest es un proyecto enfocado en el desarrollo y mejora de un sistema tecnológico para la evaluación de la estabilidad del individuo a partir de su marcha, con el objetivo de asistir en el diagnóstico clínico y contribuir a la prevención de caídas. La motivación surge del significativo impacto sanitario y económico asociado a estos eventos, especialmente en poblaciones vulnerables, así como de la necesidad de contar con herramientas objetivas, accesibles y no invasivas que permitan su detección temprana.

El trabajo se basa en un desarrollo previo realizado por la empresa Serelabs, sobre el cual se introdujeron mejoras en el procesamiento de señales, los algoritmos de análisis y la presentación de resultados, con el fin de acercar la solución a su aplicación en entornos clínicos reales.

## Sistema propuesto

El sistema desarrollado integra software de procesamiento con una unidad de medición inercial (IMU) colocada en la región lumbar del individuo, equipada con sensores como acelerómetros y giroscopios. Esta unidad registra señales asociadas al movimiento durante la marcha, las cuales son posteriormente procesadas para extraer información relevante sobre el comportamiento dinámico del paciente y permitir la inferencia de su estabilidad.

Este enfoque permite generar métricas objetivas y reproducibles que complementan la evaluación clínica, reduciendo la dependencia de observaciones subjetivas.

## Análisis de la marcha

En este contexto, la marcha se modela como un proceso cíclico compuesto por las fases de stance y swing, a partir del cual se estiman parámetros espacio-temporales tales como la cadencia, la duración del paso, la longitud del paso y la velocidad de marcha.

Para ello, se desarrollaron algoritmos de detección de eventos fundamentales del ciclo de marcha, como los contactos iniciales (heel strikes) y terminales (toe offs), que permiten segmentar las señales y calcular las métricas correspondientes. Asimismo, se emplearon modelos biomecánicos simplificados, como el modelo de péndulo invertido, para la estimación de la longitud del paso, junto con estrategias de ajuste orientadas a mejorar la precisión de las estimaciones.

Adicionalmente, se incorporan mecanismos para la detección de giros durante la marcha, basados en el análisis de la velocidad angular medida por el giroscopio. Esto permite identificar y aislar estos eventos del análisis principal, evitando que afecten la estimación de los parámetros y mejorando la robustez del sistema frente a condiciones no ideales.

## Validación experimental

La validación del sistema se llevó a cabo mediante ensayos en condiciones controladas, con pasos de longitud fija, y en condiciones no controladas, de marcha libre.

Los resultados mostraron que los parámetros temporales, como la cadencia y la duración del paso, se estimaron de manera consistente entre los distintos métodos implementados. En contraste, la estimación de la longitud del paso presentó mayor variabilidad, lo que evidencia su sensibilidad a los modelos utilizados y a las diferencias individuales entre los sujetos.

En conjunto, los resultados indican que el sistema es robusto para la estimación de parámetros temporales y presenta un desempeño prometedor en la estimación de parámetros espaciales, sujeto a calibración y mejoras adicionales.

## Presentación de resultados

El sistema incluye la generación automática de reportes clínicos en formato PDF, en los que se presentan los parámetros de marcha estimados junto con su comparación respecto a rangos de referencia obtenidos de estudios previos.

Asimismo, los resultados se complementan mediante representaciones gráficas que permiten visualizar la evolución temporal de los parámetros, así como su comparación con valores de referencia considerados normales. Esto facilita la interpretación por parte de profesionales de la salud, proporcionando una herramienta clara, estandarizada y orientada a la evaluación del estado funcional del paciente.

## Análisis tiempo-frecuencia

Dado el carácter no estacionario de las señales registradas, se incorporan técnicas de análisis tiempo-frecuencia, en particular la transformada wavelet continua (CWT). Esta herramienta permite analizar simultáneamente la evolución temporal y espectral de las señales, facilitando la identificación de patrones característicos de la marcha.

Las representaciones obtenidas en el dominio tiempo-escala, en forma de escalogramas, constituyen una base informativa rica que permite capturar tanto eventos transitorios como variaciones dinámicas del movimiento.

## Clasificación de la estabilidad del individuo

A partir de las representaciones tiempo-frecuencia obtenidas, se emplean modelos de tipo autoencoder para aprender una representación comprimida de los patrones de marcha. Este proceso permite reducir la dimensionalidad de los datos conservando la información más relevante contenida en los escalogramas.

Las representaciones latentes generadas por el autoencoder se utilizan como entrada para modelos de aprendizaje automático, mediante los cuales se aborda la clasificación de la estabilidad del individuo a partir de las señales de marcha. De este modo, la evaluación se realiza directamente sobre descriptores aprendidos a partir de estos datos, en lugar de depender exclusivamente de parámetros espacio-temporales tradicionales.

Este enfoque permite capturar relaciones complejas en la dinámica del movimiento que no son evidentes mediante métricas convencionales. Si bien los resultados obtenidos son preliminares y dependen de la disponibilidad de datos para entrenamiento y validación, el método demuestra la viabilidad de utilizar representaciones aprendidas para la evaluación automática de la estabilidad.

## Análisis de largo plazo

El proyecto contempla el análisis de registros de larga duración, para lo cual se implementaron algoritmos de detección de actividad que permiten segmentar automáticamente distintos tipos de comportamiento a partir de las señales registradas.

Sobre esta base, se posibilitó el análisis no solo de la marcha, sino también de otras actividades del individuo en contextos no controlados, lo que permite la identificación de patrones atípicos o desviaciones en su comportamiento cotidiano. Este enfoque amplía el alcance del sistema desde evaluaciones puntuales hacia el monitoreo continuo, proporcionando una caracterización más representativa del estado funcional en condiciones reales.

## Conclusiones

El proyecto demuestra la viabilidad de utilizar sensores inerciales combinados con técnicas de procesamiento de señales, análisis tiempo-frecuencia e inteligencia artificial para la evaluación de la estabilidad del individuo a partir de su marcha.

Se destaca particularmente la incorporación de estrategias de clasificación basadas en representaciones latentes, que permiten abordar la estabilidad como un problema integral y centrado en el individuo.

Si bien se identifican limitaciones —como la necesidad de mayor cantidad de datos, la mejora en la estimación de ciertos parámetros y la validación frente a sistemas de referencia más precisos—, el trabajo establece bases sólidas para el desarrollo de una herramienta no invasiva, portable y potencialmente aplicable en entornos clínicos reales.

Asimismo, abre líneas futuras de investigación en el análisis de largo plazo, la personalización de modelos y la incorporación de técnicas más avanzadas de aprendizaje automático.