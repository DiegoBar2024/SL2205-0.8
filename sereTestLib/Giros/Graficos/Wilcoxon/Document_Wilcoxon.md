# Matrices de significancia mediante test de Wilcoxon

## Descripción general

Las matrices de Wilcoxon permiten visualizar de forma compacta si existen diferencias estadísticamente significativas entre grupos para cada una de las features analizadas.

En este análisis se utiliza el test de Wilcoxon rank-sum (equivalente al test de Mann–Whitney U para muestras independientes) para comparar la distribución de cada feature entre dos grupos etarios:

- Grupo 0: sujetos con edad entre 0 y 75 años.
- Grupo 1: sujetos con edad mayor o igual a 75 años.

El objetivo es identificar qué variables presentan diferencias sistemáticas entre adultos menores de 75 años y adultos mayores de 75 años.

## Construcción de las matrices

Para cada feature se construye una matriz de comparación entre grupos.

Cada elemento de la matriz representa el resultado de una comparación estadística entre dos grupos.

En el caso de dos grupos etarios, la matriz representa:

- Comparación entre grupo 0-75 años y grupo ≥75 años.

El valor mostrado corresponde a la significancia estadística obtenida mediante el test de Wilcoxon.

## Codificación visual

Las matrices utilizan una representación binaria:

- Valor 1:
    - Existe diferencia estadísticamente significativa entre los grupos.
    - El p-valor obtenido es menor al nivel de significancia definido (α = 0.05).

- Valor 0:
    - No existe evidencia estadística suficiente para afirmar que las distribuciones sean diferentes.
    - El p-valor es mayor o igual al nivel de significancia.

- Valores NaN:
    - Corresponden a comparaciones no definidas, como la comparación de un grupo consigo mismo.

Cuando se utiliza corrección por múltiples comparaciones mediante FDR (False Discovery Rate), la decisión de significancia se basa en el p-valor corregido.

## Interpretación del gráfico

Cada matriz corresponde a una única feature.

Una celda marcada como significativa indica que la distribución de esa variable presenta diferencias entre los grupos etarios comparados.

Por ejemplo:

- Una feature con diferencia significativa entre 0-75 años y ≥75 años indica que sus valores tienden a cambiar con la edad.

- Una feature sin diferencias significativas indica que no se observa una modificación clara de esa característica entre ambos grupos.

## Relación con envejecimiento

Las matrices permiten identificar qué variables presentan cambios asociados al envejecimiento.

Una alta cantidad de features significativas puede indicar que existen modificaciones sistemáticas en la señal o en el comportamiento motor asociado a edades avanzadas.

Sin embargo, la significancia estadística no implica necesariamente capacidad predictiva.

Una feature puede mostrar diferencias estadísticamente significativas entre grupos, pero dichas diferencias pueden ser pequeñas y no suficientes para clasificar correctamente nuevos sujetos.

## Relación con el ranking SVM

Las matrices de Wilcoxon y el ranking SVM evalúan aspectos complementarios:

### Wilcoxon

Responde principalmente a:

> ¿Existe una diferencia estadísticamente significativa entre las distribuciones de los grupos?

### SVM univariado

Responde principalmente a:

> ¿Qué tan bien permite esta feature separar individuos pertenecientes a distintos grupos?

Por lo tanto:

- Una feature idealmente relevante debería presentar diferencias significativas mediante Wilcoxon y bajo error de clasificación mediante SVM.

- La combinación de ambos análisis permite identificar variables con evidencia estadística y potencial utilidad predictiva.

## Consideraciones finales

El análisis mediante Wilcoxon debe interpretarse considerando:

- El tamaño de muestra disponible en cada grupo.

- La variabilidad interna de las medidas.

- La corrección por múltiples comparaciones cuando se evalúan muchas features simultáneamente.

Estas matrices constituyen una herramienta exploratoria para identificar variables potencialmente asociadas con diferencias relacionadas con la edad, mientras que el ranking SVM permite evaluar su utilidad práctica para discriminación automática entre grupos.