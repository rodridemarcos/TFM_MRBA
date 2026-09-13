# TFM_MRBA

- Entorno TFM:
  El proyecto ha sido desarrollado en Python utilizando principalmente
  Spyder y Google Colab (o en su defecto Jupyter Notebook).
  
  Para reproducir el entorno necesario para la ejecución del código,
  se recomienda crear un entorno virtual e instalar las dependencias
  incluidas en `Entorno_TFM.txt`:
  
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r Entorno_TFM.txt

- Carpeta "construcción_dataset":
  Esta carpeta contiene dos subcarpetas correspondientes a las dos etapas de procesamiento de los datos:
  1) Subcarpeta "dataset_maestro":
     - Codigo_dataset_maestro.py: Script que procesa las series originales ubicadas en la carpeta "fuentes" para generar un conjunto de datos                    inicial homogeneizado de 127 variables (dataset_maestro_vivienda_trimestral_2015_2024_IAV.csv).
     - Contiene además el archivo de auditoría del conjunto maestro (auditoria_dataset_maestro.csv), que describe el contenido y la cobertura de cada una        de las variables.
  2) Subcarpeta "dataset_TFM_MRBA":
     - construir_dataset_TFM_MRBA.py: Script que toma como entrada el archivo dataset_maestro_vivienda_trimestral_2015_2024_IAV.csv y el fichero                 fuentes_complementarias_provinciales.csv para generar el conjunto de datos definitivo empleado en la modelización de este TFM.
     - Subcarpeta "salidas": Ubicación donde el script almacena el dataset final (dataset_TFM_MRBA.csv) junto con su correspondiente fichero de auditoría        (auditoria_dataset_TFM_MRBA.csv), en el que se detallan las descripciones, unidades y características de las 37 variables seleccionadas.
 
- Carpeta "modelización":
  En esta carpeta se encuentra el código, realizado en Google Colab, que analiza el dataset creado previamente. Se incluye en formato .ipynb y .html,       comentado en su totalidad de forma más técnica y a partir del cual se ha redactado la memoria del TFM.

- Ejecución de la aplicación:
  Descargar carpeta de "app" y subirla al entorno de desarrollo o IDE donde se vaya a ejecutar (en este caso, Spyder).
  1) Descargar esta carpeta completa (es el zip que se genera al final del apartado 6 descomprimido) y abrir el script principal y establecer como             directorio de trabajo (working directory) la ruta donde se encuentran estos datos que usará la aplicación.
  2) Ejecutar el código y copiar la URL local http://127.0.0.1:5000 en el navegador web.
  3) Interactuar con la aplicación.
  4) Para detener la ejecución, pulsar Ctrl + C en la consola de Spyder y cerrar la pestaña en el navegador.
