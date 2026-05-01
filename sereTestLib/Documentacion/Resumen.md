# Resumen del Proyecto: Seretest

## Introducción y motivación

Seretest fue un proyecto que se centró en el desarrollo y la mejora de un sistema tecnológico orientado a la evaluación de la estabilidad de la marcha humana, con el objetivo de asistir en el diagnóstico clínico y la prevención de caídas. La motivación principal surgió del impacto sanitario y económico que estas generan, especialmente en poblaciones vulnerables, lo que puso de manifiesto la necesidad de contar con herramientas confiables, accesibles y no invasivas que permitieran su detección temprana.

El trabajo se basó en un desarrollo previo realizado por la empresa Serelabs, sobre el cual se introdujeron mejoras tanto en el procesamiento de datos como en los algoritmos de análisis y en la presentación de resultados, con el fin de acercar la solución a un contexto clínico real.

## Sistema propuesto

El sistema desarrollado utilizó una implementación de software junto a unidades de medición inercial (IMU), que integran sensores como acelerómetros y giroscopios, colocadas en la región lumbar del individuo. Estos sensores registraron señales asociadas al movimiento corporal durante la marcha, las cuales fueron posteriormente procesadas mediante software para extraer información relevante sobre la estabilidad y el comportamiento del paciente durante la marcha.

A partir de estas mediciones, el sistema buscó generar información objetiva que complementara la evaluación médica, reduciendo la dependencia de observaciones subjetivas y permitiendo un análisis más sistemático y reproducible.

## Análisis de la marcha

Uno de los aspectos fundamentales del proyecto fue el análisis de la marcha humana, entendida como un proceso cíclico compuesto por las fases de $\textit{stance}$ y $\textit{swing}$. En base a este análisis, se estimaron parámetros de marcha como la cadencia (número de pasos por unidad de tiempo), la duración del paso, la longitud del paso y la velocidad de marcha.

Para calcular estos parámetros, se desarrollaron algoritmos capaces de identificar eventos fundamentales del ciclo de marcha, como los contactos iniciales ($\textit{heel strikes}$) y los contactos terminales ($\textit{toe offs}$). A partir de estos eventos, se segmentaron las señales registradas y se calcularon las métricas correspondientes. Se implementaron distintos enfoques, incluyendo modelos biomecánicos simplificados, como el modelo de péndulo invertido, junto con técnicas de procesamiento de señales que permitieron adaptar el análisis a las características de cada registro.

## Procesamiento de señales y análisis tiempo-frecuencia

Dado que las señales obtenidas de los sensores presentan un comportamiento no estacionario, es decir, variable en el tiempo, el proyecto incorporó herramientas de análisis tiempo-frecuencia, en particular la transformada wavelet continua (CWT). Esta técnica permitió analizar simultáneamente la evolución temporal y espectral de las señales, facilitando la identificación de patrones característicos de la marcha que no son fácilmente detectables en el dominio temporal o frecuencial por separado.

La representación en el dominio tiempo-escala obtenida mediante CWT constituyó una base rica en información, que permitió capturar tanto eventos transitorios como variaciones en la dinámica de la marcha a lo largo del tiempo.

## Clasificación y detección de anomalías

Sobre estas representaciones tiempo-frecuencia se aplicaron modelos de inteligencia artificial con el objetivo de clasificar a las personas en función de su estabilidad. Se emplearon tanto clasificadores supervisados como modelos no supervisados, tales como autoencoders, que permitieron aprender patrones normales de movimiento y detectar desviaciones respecto de estos.

Este enfoque permitió avanzar desde una simple estimación de parámetros hacia un análisis más profundo, en el cual el sistema no solo describió la marcha, sino que también fue capaz de inferir posibles alteraciones en la estabilidad del paciente.

## Análisis de largo plazo

Un aspecto relevante del proyecto fue la consideración de registros de marcha de larga duración. A diferencia de los análisis tradicionales basados en sesiones cortas, el análisis de largo plazo permitió observar la evolución de la marcha y otras actividades en el tiempo y detectar cambios progresivos o patrones que no se manifiestan en intervalos breves.

La combinación de técnicas de procesamiento de señales, análisis tiempo-frecuencia y modelos de aprendizaje automático abrió la posibilidad de monitorear pacientes de forma continua, proporcionando información más completa y representativa de su estado funcional.

## Detección de giros

Adicionalmente, se desarrolló un algoritmo para la detección de giros durante la marcha, basado en el análisis de la velocidad angular medida por el giroscopio. Esto permitió identificar y aislar estos eventos del análisis principal, evitando que afectaran la estimación de los parámetros de marcha.

Asimismo, los giros pudieron analizarse de forma independiente, ya que su ejecución puede aportar información relevante sobre la estabilidad y el control motor del individuo.

## Validación experimental

La validación del sistema se llevó a cabo mediante pruebas experimentales en las que los participantes caminaron tanto en condiciones controladas, con pasos de longitud fija, como en condiciones naturales. A partir de estos ensayos, se compararon los resultados obtenidos con los algoritmos desarrollados frente a métodos existentes.

Los resultados mostraron que los parámetros temporales, como la cadencia y la duración del paso, se estimaron de manera consistente entre métodos. Sin embargo, la estimación de la longitud del paso presentó mayores variaciones, lo cual se explicó por la sensibilidad del modelo utilizado y por las diferencias individuales entre los participantes.

## Presentación de resultados

Un aspecto destacado del proyecto fue la automatización en la generación de reportes clínicos. El sistema produjo informes en formato PDF que incluyeron los parámetros de marcha calculados y su comparación con rangos considerados normales, basados en estudios clínicos previos.

Esto se realizó con el propósito de facilitar la interpretación por parte de profesionales de la salud, ofreciendo una herramienta clara, estandarizada y de fácil uso en contextos clínicos, lo cual representó un paso importante hacia la aplicabilidad práctica del sistema.

## Conclusiones

En conclusión, el proyecto representó un avance significativo hacia el desarrollo de una herramienta no invasiva para la evaluación de la estabilidad de la marcha. Se demostró la viabilidad de utilizar sensores inerciales junto con técnicas avanzadas de procesamiento de señales, análisis tiempo-frecuencia e inteligencia artificial para estimar parámetros relevantes y detectar posibles anomalías, así como para la clasificación de la estabilidad.

No obstante, también se identificaron ciertas limitaciones, como la necesidad de contar con una mayor cantidad de datos para validar los modelos, mejorar la precisión en la estimación de algunos parámetros y contrastar los resultados con sistemas de referencia más precisos, como los sistemas de captura de movimiento.

Finalmente, el trabajo sentó las bases para futuras líneas de investigación, incluyendo el análisis de registros de largo plazo, la personalización de los modelos para cada paciente y la incorporación de técnicas más avanzadas de inteligencia artificial, con el objetivo de lograr una herramienta robusta y plenamente aplicable en entornos clínicos reales.