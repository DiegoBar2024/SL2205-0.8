# SERETEST: CARACTERIZACIÓN DE GIROS

## CARACTERIZACIÓN DE LA POBLACIÓN SEGÚN SU FRANJA ETARIA

En principio, la intención es dividir a la población en cuatro categorías principales según su franja etaria: 0-45 años, 45-60 años, 60-75 años, mayor a 75 años.

<p align="center">
  <img src="CaractEtariaTotal.png" alt="My image" style="max-width: 60%;"><br>
  <strong>Figura:</strong> Gráfico de barras mostrando la distribución de los datos organizados por franja etaria, de todos los pacientes de los que se tiene información. No se cuentan con registros de todos los pacientes.
</p>

<p align="center">
  <img src="CaractEtaria.png" alt="My image" style="max-width: 60%;"><br>
  <strong>Figura:</strong> Gráfico de barras mostrando la distribución de los datos organizados por franja etaria, de aquellos pacientes para los que se cuentan con registros en la base de datos
</p>

## DETECCIÓN DE GIROS

### DIAGRAMA DE FLUJO DEL ALGORITMO

<p align="center">
  <img src="PipelineGiros.png" alt="My image" style="max-width: 60%;"><br>
  <strong>Figura:</strong> Diagrama de flujo que ilustra en alto nivel el proceso de alineación de la velocidad angular con un sistema inercial ENU/NED para luego calcular los giros.
</p>

### PRUEBAS SOBRE REGISTROS DE MARCHA

El sistema de referencia que estoy usando para calcular la orientación en estas pruebas es 'ENU' (East North Up).

<p align="center">
  <img src="GirosRodrigoEstandar.png" alt="My image" style="max-width: 80%;"><br>
  <strong>Figura:</strong> Gráfico de la componente vertical de la velocidad angular indicando con rojo los intervalos en los que se detectan giros. Registro <code>'MarchaEstandar_Rodrigo.txt'</code>
</p>

<p align="center">
  <img src="GirosSabrinaEstandar.png" alt="My image" style="max-width: 80%;"><br>
  <strong>Figura:</strong> Gráfico de la componente vertical de la velocidad angular indicando con rojo los intervalos en los que se detectan giros. Registro <code>'MarchaEstandar_Sabrina.txt'</code>
</p>

<p align="center">
  <img src="GirosRodriLibre.png" alt="My image" style="max-width: 80%;"><br>
  <strong>Figura:</strong> Gráfico de la componente vertical de la velocidad angular indicando con rojo los intervalos en los que se detectan giros. Registro <code>'MarchaLibre_Rodrigo.txt'</code>
</p>

<p align="center">
  <img src="GirosSabrinaLibre.png" alt="My image" style="max-width: 80%;"><br>
  <strong>Figura:</strong> Gráfico de la componente vertical de la velocidad angular indicando con rojo los intervalos en los que se detectan giros. Registro <code>'MarchaLibre_Sabrina.txt'</code>
</p>
