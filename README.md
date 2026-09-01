# TFM_MRBA

- Carpeta "construcción_dataset":
  En esta carpeta se encuentran varios códigos y conjuntos de datos.
  1) Codigo_dataset_modelado_final.py: este código usa las series de datos que se encuentran en la carpeta "fuentes" para crear un dataset con 127 variables homogeneizadas llamado dataset_maestro_vivienda_trimestral_2015_2024_IAV.csv.
  2) construir_dataset_TFM_MRBA.py: este código usa tanto dataset_maestro_vivienda_trimestral_2015_2024_IAV.csv como el de fuentes_complementarias_provinciales.csv para desarrollar el dataset_TFM_MRBA para su posterior análisis en este TFM. Este script también genera un fichero de auditoría, que muestra la descripción de las distintas variables que componen el dataset.
 
- Carpeta "modelización":
  En esta carpeta se encuentra el código, realizado en google colab, que analiza el dataset creado previamente, en formato .ipynb y .html, comentado en su totalidad de forma técnica.

- Ejecución de la aplicación:
  Descargar carpeta de "app" y subirla al entorno de desarrollo o IDE donde se vaya a ejecutar (en este caso, Spyder).
  1) Descargar esta carpeta completa (es el zip que se genera al final del apartado 6 descomprimido) y abrir el script principal y establecer como directorio de trabajo (working directory) la ruta donde se encuentran estos datos que usará la aplicación.
  2) Ejecutar el código y copiar la URL local http://127.0.0.1:5000 en el navegador web.
  3) Interactuar con la aplicación.
  4) Para detener la ejecución, pulsar Ctrl + C en la consola de Spyder y cerrar la pestaña en el navegador.
