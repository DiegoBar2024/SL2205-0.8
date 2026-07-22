# Ranking de features más discriminativas utilizando SVM univariado

## Descripción general

Este gráfico representa un ranking de las variables (features) con mayor capacidad para diferenciar dos grupos de interés mediante un clasificador SVM (Support Vector Machine) aplicado de manera univariada.

En este análisis, cada feature es evaluada individualmente, es decir, se entrena un modelo SVM utilizando únicamente una variable a la vez como entrada. El objetivo es determinar qué características presentan mayor capacidad discriminativa entre los grupos comparados.

Para este caso, la separación analizada corresponde a dos grupos etarios:

- Grupo 0: sujetos con edad entre 0 y 75 años.
- Grupo 1: sujetos con edad mayor o igual a 75 años.

Cada feature representa una característica extraída de las señales analizadas (por ejemplo, características temporales, frecuenciales o relacionadas con la dinámica del movimiento durante los giros).

## Construcción del gráfico

Para cada feature individual se realiza el siguiente procedimiento:

1. Se selecciona una única variable como entrada del modelo SVM.

2. Se entrenan clasificadores SVM independientes utilizando únicamente dicha feature.

3. Se evalúa el rendimiento del clasificador mediante el error de clasificación obtenido.

4. Las features se ordenan según dicho error, generando un ranking de capacidad discriminativa.

El eje X representa las diferentes features evaluadas.

El eje Y representa el error de clasificación obtenido por el SVM univariado.

## Interpretación del ranking

La interpretación principal del gráfico se basa en que:

- Un menor error de clasificación indica una mayor capacidad de la feature para separar los dos grupos etarios.

- Una feature ubicada en las primeras posiciones del ranking presenta una mayor diferencia entre las distribuciones de ambos grupos.

- Una feature ubicada en posiciones inferiores presenta menor capacidad para discriminar entre sujetos menores de 75 años y sujetos de 75 años o más.

Por lo tanto:

- Features con error cercano a 0 indican una separación potencialmente fuerte entre grupos.

- Features con error cercano al azar (aproximadamente 0.5 para clasificación binaria balanceada) tienen poca capacidad discriminativa.

## Consideraciones importantes

El ranking obtenido corresponde a un análisis univariado. Esto significa que:

- Cada feature es evaluada de forma independiente.

- No se consideran combinaciones entre múltiples variables.

- Una feature con buen desempeño individual no necesariamente será la más importante dentro de un modelo multivariado.

El objetivo principal de este análisis es realizar una primera exploración de cuáles variables contienen información asociada al envejecimiento o a diferencias relacionadas con la edad.

## Relación con análisis estadístico

El rendimiento predictivo del SVM puede complementarse con pruebas estadísticas como el test de Wilcoxon.

Una feature puede presentar:

- Bajo error SVM y diferencia estadísticamente significativa entre grupos, indicando una separación consistente.

- Bajo error SVM pero sin significancia estadística, lo cual puede deberse a efectos relacionados con el tamaño muestral o variabilidad interna.

- Significancia estadística pero bajo poder predictivo, indicando que existen diferencias entre grupos, aunque no sean suficientes para clasificar correctamente nuevos individuos.

Por este motivo, el ranking SVM debe interpretarse junto con los análisis estadísticos complementarios.