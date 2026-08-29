"""
================================================================================
CONSTRUCCIÓN DEL DATASET DEFINITIVO DEL TFM
Predicción de la accesibilidad a la vivienda en España (2015-2024)

Autor: Marcos Rodrigo Bermejo Arroyo
Máster en Big Data, Data Science & Artificial Intelligence

--------------------------------------------------------------------------------
CADENA DE PROCEDENCIA DE LOS DATOS

  Etapa 1. Descarga y depuración de las fuentes oficiales (INE, AEAT, Banco de
           España y Ministerio de Vivienda), organizadas por dominio temático en
           las siete carpetas del repositorio.
             -> Codigo_dataset_modelado_final.py

  Etapa 2. Integración de las fuentes en un conjunto maestro de 127 variables
           con estructura de panel provincial trimestral.
             -> dataset_maestro_vivienda_trimestral_2015_2024_IAV.csv

  Etapa 3. Selección de variables, reconstrucción del precio de la vivienda y
           cálculo del indicador de accesibilidad.
             -> este script
             -> dataset_TFM_MRBA.csv + auditoria_dataset_TFM_MRBA.csv

--------------------------------------------------------------------------------
FUENTES ORIGINALES

  Instituto Nacional de Estadística
    - Índice de Precios de Vivienda (IPV), por comunidad autónoma
    - Índice de Precios al Consumo, por provincia
    - Encuesta de Población Activa
    - Atlas de Distribución de Renta de los Hogares
    - Cifras de Población y Encuesta Continua de Hogares
    - Estadística de Transmisiones de Derechos de la Propiedad
    - Estadística de Hipotecas
    - Estadística de Migraciones
    - Coyuntura Turística Hotelera

  Agencia Estatal de Administración Tributaria
    - Mercado de trabajo y pensiones en las fuentes tributarias.
      Salario medio anual por provincia y sexo. Los territorios de régimen
      foral no figuran en esta estadística y proceden de sus respectivas
      haciendas forales.

  Banco de España
    - Tipos de interés de los préstamos para adquisición de vivienda

  Ministerio de Vivienda y Agenda Urbana
    - Valor tasado de la vivienda libre
    - Índice de costes de construcción
    - Visados de obra nueva
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# 0. PARÁMETROS
# ==============================================================================

# Las fuentes se leen del repositorio del proyecto. Para trabajar sin conexión
# basta con descargar los archivos en DIR_LOCAL y cambiar USAR_REPOSITORIO.
USAR_REPOSITORIO = True

URL_BASE = (
    "https://raw.githubusercontent.com/rodridemarcos/TFM_MRBA/refs/heads/main/"
)

RUTAS_FUENTES = {
    "maestro": "Vivienda/dataset_maestro_vivienda_trimestral_2015_2024_IAV.csv",
    "adicional": "Vivienda/fuentes_complementarias_provinciales.csv",
    "renta": "Econom%C3%ADa/renta_provincia_anual.csv",
}

DIR_LOCAL = Path("entradas")
DIR_SALIDA = Path("salidas")

ANIO_BASE = 2015                 # Año de anclaje del nivel de precios
SUPERFICIE_REFERENCIA_M2 = 80    # Vivienda tipo del indicador
DECIMALES = 3


def ruta(clave):
    """Devuelve la ubicación de una fuente, remota o local."""
    if USAR_REPOSITORIO:
        return URL_BASE + RUTAS_FUENTES[clave]
    return DIR_LOCAL / Path(RUTAS_FUENTES[clave]).name


# ==============================================================================
# 1. LECTURA DE LAS FUENTES
# ==============================================================================

def leer_fuentes():
    """Carga el dataset maestro, los salarios de la AEAT y la renta del INE."""
    maestro = pd.read_csv(
        ruta("maestro"), sep=",", decimal=".", encoding="utf-8-sig",
    )
    maestro.columns = maestro.columns.str.strip()
    maestro["periodo"] = (
        maestro["anio"].astype(str) + "T" + maestro["trimestre"].astype(str)
    )

    # Fuentes provinciales que no figuran en el maestro: los salarios de la
    # AEAT, el IPC provincial en nivel, la estructura de edad intermedia, la
    # migración, las compraventas, las hipotecas y los visados de obra nueva.
    adicional = pd.read_csv(
        ruta("adicional"), sep=";", decimal=",", encoding="utf-8-sig",
    )
    adicional.columns = adicional.columns.str.strip()

    renta = pd.read_csv(
        ruta("renta"), sep=",", decimal=".", encoding="utf-8-sig",
    )

    return maestro, adicional, renta


# ==============================================================================
# 2. INTEGRACIÓN DE LAS FUENTES
# ==============================================================================

def integrar(maestro, adicional, renta):
    """Une las tres fuentes sobre la clave provincia-periodo."""

    # --- Salarios de la AEAT y variables provinciales del info_adicional ---
    columnas_adicionales = [
        "provincia", "periodo",
        "salario_medio_anual_euros_total",
        "salario_medio_anual_euros_hombre",
        "salario_medio_anual_euros_mujer",
        "indice_precios_consumo_base_2015",
        "pct_35_64",
        "tasa_migracion_neta_exterior_por_1000_habitantes",
        "compraventas_viviendas",
        "pct_compraventas_vivienda_nueva",
        "importe_medio_hipoteca_vivienda_eur",
        "visados_obra_nueva_viviendas",
    ]

    df = maestro.drop(
        columns=[
            c for c in [
                "salario_medio_anual_euros_total",
                "salario_medio_anual_euros_hombre",
                "salario_medio_anual_euros_mujer",
                "tasa_migracion_neta_exterior_por_1000_habitantes",
            ]
            if c in maestro.columns
        ]
    ).merge(
        adicional[columnas_adicionales],
        on=["provincia", "periodo"],
        how="left",
        validate="one_to_one",
    )

    # --- Renta de los hogares (anual, se replica en los cuatro trimestres) ---
    # El maestro arrastra una versión parcial de estas series, por lo que se
    # eliminan antes de incorporar la fuente original del Atlas de Renta.
    columnas_renta = [c for c in renta.columns if c not in ("provincia", "anio")]
    df = df.drop(columns=[c for c in columnas_renta if c in df.columns])

    df = df.merge(renta, on=["provincia", "anio"], how="left", validate="many_to_one")

    return df.copy()


# ==============================================================================
# 3. RECONSTRUCCIÓN DEL PRECIO DE LA VIVIENDA
# ==============================================================================

def reconstruir_precio(df):
    """Reancla el nivel provincial de 2015 con la evolución del IPV del INE.

    El valor tasado del Ministerio subestima de forma sistemática la evolución
    del mercado: entre 2015 y 2024 crece un 18,8 % frente al 47,7 % del Índice
    de Precios de Vivienda del INE, porque las tasaciones se apoyan en
    comparables pasados y responden con retraso.

    Se combinan por tanto las dos fuentes según su fortaleza respectiva: el
    valor tasado aporta el NIVEL provincial de 2015, que es fiable para comparar
    territorios entre sí, y el IPV aporta la EVOLUCIÓN, que es la magnitud
    oficial de referencia.

        precio_m2[p,t] = valor_tasado[p, 2015] × IPV[p,t] / IPV[p, 2015]

    Limitación: el IPV se publica por comunidad autónoma, de modo que las
    provincias de una misma comunidad comparten tasa de variación, aunque
    conservan niveles distintos.
    """
    base = (
        df.loc[df["anio"] == ANIO_BASE]
        .groupby("provincia")
        .agg(
            valor_tasado_base=("valor_tasado_m2", "mean"),
            ipv_base=("ipv_general", "mean"),
        )
        .reset_index()
    )

    df = df.merge(base, on="provincia", how="left", validate="many_to_one")

    df["precio_m2_mercado"] = (
        df["valor_tasado_base"] * df["ipv_general"] / df["ipv_base"]
    )

    return df.drop(columns=["valor_tasado_base", "ipv_base"])


# ==============================================================================
# 4. CÁLCULO DEL INDICADOR DE ACCESIBILIDAD
# ==============================================================================

def calcular_iav(df):
    """Años de salario bruto necesarios para adquirir una vivienda de 80 m².

    El indicador es un cociente entre dos magnitudes nominales, por lo que la
    inflación se cancela por construcción y NO debe aplicarse ningún ajuste
    adicional de precios sobre el numerador.
    """
    coste = SUPERFICIE_REFERENCIA_M2 * df["precio_m2_mercado"]

    df["IAV_total"] = coste / df["salario_medio_anual_euros_total"]
    df["IAV_hombre"] = coste / df["salario_medio_anual_euros_hombre"]
    df["IAV_mujer"] = coste / df["salario_medio_anual_euros_mujer"]
    df["brecha_iav_absoluta"] = df["IAV_mujer"] - df["IAV_hombre"]

    return df


# ==============================================================================
# 5. VARIABLES DERIVADAS
# ==============================================================================

def crear_derivadas(df):
    """Construye tasas relativas a población y variables de presión turística.

    Las magnitudes absolutas ligadas al tamaño del territorio presentan una
    asimetría muy acusada, ya que unas pocas provincias concentran la mayor
    parte de la actividad. Expresarlas por cada 1.000 habitantes las hace
    comparables entre territorios.
    """
    poblacion_miles = df["poblacion_total"] / 1000

    df["compraventas_por_1000_hab"] = df["compraventas_viviendas"] / poblacion_miles
    df["visados_obra_nueva_por_1000_hab"] = (
        df["visados_obra_nueva_viviendas"] / poblacion_miles
    )
    df["plazas_hoteleras_por_1000_hab"] = df["plazas_estimadas"] / poblacion_miles
    df["pernoctaciones_por_habitante"] = (
        df["pernoctaciones_total"] / df["poblacion_total"]
    )

    return df


# ==============================================================================
# 6. ARRASTRE DE LAS SERIES DE PUBLICACIÓN TARDÍA
# ==============================================================================

def arrastrar_series_anuales(df, columnas):
    """Propaga el último valor publicado en las series anuales incompletas.

    El Atlas de Distribución de Renta y las estadísticas de estructura
    demográfica del INE se publican con dos años de retraso, por lo que en 2024
    no existe dato disponible. Se propaga el último valor conocido de cada
    provincia, que es exactamente la información de la que dispondría una
    aplicación en producción en ese momento.

    No se trata por tanto de una imputación estadística, sino de la
    reproducción del calendario real de publicación. Se marca cada observación
    afectada para poder evaluar después si el error se concentra en ellas.

    El Atlas de Distribución de Renta no cubrió inicialmente los territorios de
    régimen foral, por lo que Araba/Álava, Gipuzkoa y Navarra carecen de dato en
    los primeros años. Para ese hueco inicial no existe valor anterior que
    propagar, de modo que se retrocede el primer valor publicado. Esta operación
    afecta únicamente a los años iniciales y nunca al periodo de prueba.
    """
    df = df.sort_values(["provincia", "anio", "trimestre"]).reset_index(drop=True)

    faltaba = df[columnas].isna().any(axis=1)

    por_provincia = df.groupby("provincia", group_keys=False)[columnas]
    df[columnas] = por_provincia.ffill()

    por_provincia = df.groupby("provincia", group_keys=False)[columnas]
    df[columnas] = por_provincia.bfill()

    df["valores_arrastrados"] = (faltaba & df[columnas].notna().all(axis=1)).astype(int)

    return df


# ==============================================================================
# 7. SELECCIÓN DE VARIABLES
# ==============================================================================

# Cada entrada define: bloque, descripción y tipo de valor.
# El nivel de desagregación se calcula automáticamente en la auditoría.

VARIABLES_FINALES = {
    # ---------------------------- Identificadores -----------------------------
    "periodo": ("Identificadores", "Año y trimestre de la observación", "Categórico"),
    "codigo_provincia": ("Identificadores", "Código numérico del territorio", "Entero"),
    "provincia": ("Identificadores", "Provincia o ciudad autónoma", "Categórico"),
    "comunidad_autonoma": ("Identificadores", "Comunidad autónoma", "Categórico"),

    # ------------------------- Demografía y hogares ---------------------------
    "poblacion_total": ("Demografía y hogares", "Población total del territorio", "Entero"),
    "pct_18_34": ("Demografía y hogares", "Población de entre 18 y 34 años", "%"),
    "pct_35_64": ("Demografía y hogares", "Población de entre 35 y 64 años", "%"),
    "pct_65_mas": ("Demografía y hogares", "Población de 65 años o más", "%"),
    "tamano_medio_hogar": ("Demografía y hogares", "Número medio de personas por hogar", "Decimal"),
    "pct_hogares_unipersonales": ("Demografía y hogares", "Proporción de hogares unipersonales", "%"),
    "pct_poblacion_espanola": ("Demografía y hogares", "Proporción de población de nacionalidad española", "%"),
    "tasa_migracion_neta_exterior_por_1000_habitantes": (
        "Demografía y hogares", "Saldo migratorio exterior por cada 1.000 habitantes", "Tasa"),

    # ---------------------- Mercado laboral y rentas --------------------------
    "salario_medio_anual_euros_total": ("Mercado laboral y rentas", "Salario bruto medio anual", "Euros"),
    "salario_medio_anual_euros_hombre": ("Mercado laboral y rentas", "Salario bruto medio anual de los hombres", "Euros"),
    "salario_medio_anual_euros_mujer": ("Mercado laboral y rentas", "Salario bruto medio anual de las mujeres", "Euros"),
    "tasa_actividad_pct_total": ("Mercado laboral y rentas", "Tasa de actividad", "%"),
    "tasa_empleo_pct_total": ("Mercado laboral y rentas", "Tasa de empleo", "%"),
    "tasa_paro_pct_total": ("Mercado laboral y rentas", "Tasa de paro", "%"),
    "renta_neta_media_por_hogar_eur": ("Mercado laboral y rentas", "Renta neta media por hogar", "Euros"),
    "renta_mediana_por_unidad_consumo_eur": ("Mercado laboral y rentas", "Renta mediana por unidad de consumo", "Euros"),

    # ------------------------ Entorno económico -------------------------------
    "indice_precios_consumo_base_2015": (
        "Entorno económico", "Índice de precios al consumo provincial, base 2015 = 100", "Índice"),
    "indice_costes_construccion": ("Entorno económico", "Índice de costes de construcción", "Índice"),

    # ---------------------------- Financiación --------------------------------
    "tipo_interes_prestamos_vivienda_pct": ("Financiación", "Tipo de interés de los préstamos para vivienda", "%"),
    "importe_medio_hipoteca_vivienda_eur": ("Financiación", "Importe medio de las hipotecas sobre vivienda", "Euros"),

    # --------------------- Mercado residencial y oferta -----------------------
    "valor_tasado_m2": ("Mercado residencial y oferta", "Valor tasado medio de la vivienda", "Euros/m²"),
    "ipv_general": ("Mercado residencial y oferta", "Índice de precios de vivienda de la comunidad autónoma", "Índice"),
    "compraventas_por_1000_hab": ("Mercado residencial y oferta", "Compraventas trimestrales por cada 1.000 habitantes", "Tasa"),
    "pct_compraventas_vivienda_nueva": ("Mercado residencial y oferta", "Proporción de compraventas de vivienda nueva", "%"),
    "visados_obra_nueva_por_1000_hab": ("Mercado residencial y oferta", "Visados de obra nueva por cada 1.000 habitantes", "Tasa"),

    # ------------------------- Presión turística ------------------------------
    "plazas_hoteleras_por_1000_hab": ("Presión turística", "Plazas hoteleras por cada 1.000 habitantes", "Tasa"),
    "pernoctaciones_por_habitante": ("Presión turística", "Pernoctaciones hoteleras por habitante", "Tasa"),

    # -------------------------- Control de calidad ----------------------------
    "valores_arrastrados": ("Control de calidad", "Indica si la observación usa series anuales propagadas", "Binario"),

    # ------------------------- Variables derivadas ----------------------------
    "precio_m2_mercado": ("Variables derivadas", "Precio de mercado reanclado con el IPV del INE", "Euros/m²"),
    "IAV_hombre": ("Variables derivadas", "IAV calculado con el salario masculino", "Años de salario"),
    "IAV_mujer": ("Variables derivadas", "IAV calculado con el salario femenino", "Años de salario"),
    "brecha_iav_absoluta": ("Variables derivadas", "Diferencia entre el IAV femenino y el masculino", "Años de salario"),

    # --------------------------- Variable objetivo ----------------------------
    "IAV_total": ("Variable objetivo",
                  "Años de salario necesarios para adquirir una vivienda de 80 m²",
                  "Años de salario"),
}

# Variables que NO pueden emplearse como predictoras por derivar del objetivo.
NO_PREDICTORAS = [
    "precio_m2_mercado", "IAV_hombre", "IAV_mujer", "brecha_iav_absoluta",
    "IAV_total", "salario_medio_anual_euros_hombre", "salario_medio_anual_euros_mujer",
]


# ==============================================================================
# 8. AUDITORÍA
# ==============================================================================

def determinar_nivel(df, columna):
    """Clasifica una variable según su grado de desagregación territorial.

    No basta con contar valores distintos, ya que algunas series provinciales
    aparecen redondeadas y toman pocos valores diferentes. Se comprueba por
    tanto si las provincias de una misma comunidad autónoma comparten siempre
    el mismo valor, que es la condición que define una serie autonómica.
    """
    if columna in ("periodo", "provincia", "codigo_provincia", "comunidad_autonoma"):
        return "Identificador"
    if columna == "valores_arrastrados":
        return "No aplica"

    if df.groupby("periodo")[columna].nunique().mean() <= 1.05:
        return "Nacional"

    coincide_en_comunidad = (
        df.groupby(["periodo", "comunidad_autonoma"])[columna].nunique().max() == 1
    )
    return "Autonómico" if coincide_en_comunidad else "Provincial"


def determinar_uso(bloque, variable):
    """Indica el papel que puede desempeñar cada variable en la modelización."""
    if bloque == "Identificadores":
        return "Identificador"
    if bloque == "Control de calidad":
        return "Control"
    if variable in NO_PREDICTORAS:
        return "No predictora"
    return "Predictora"


def construir_auditoria(df):
    """Genera la tabla de auditoría con la estructura solicitada."""
    filas = []
    for variable, (bloque, descripcion, tipo_valor) in VARIABLES_FINALES.items():
        filas.append({
            "Clasificación": bloque,
            "Variable": variable,
            "Tipología": str(df[variable].dtype),
            "Descripción": descripcion,
            "Tipo de valor": tipo_valor,
            "Nivel de desagregación": determinar_nivel(df, variable),
            "Uso": determinar_uso(bloque, variable),
            "Valores ausentes": int(df[variable].isna().sum()),
            "Cobertura (%)": round(df[variable].notna().mean() * 100, 2),
        })

    auditoria = pd.DataFrame(filas)

    # Nº de variables por clasificación, tal y como aparece en la tabla resumen.
    auditoria.insert(
        1, "Nº",
        auditoria.groupby("Clasificación")["Variable"].transform("size"),
    )

    orden = list(dict.fromkeys(v[0] for v in VARIABLES_FINALES.values()))
    auditoria["_orden"] = auditoria["Clasificación"].map(orden.index)

    return (
        auditoria.sort_values(["_orden", "Variable"])
        .drop(columns="_orden")
        .reset_index(drop=True)
    )


# ==============================================================================
# 9. PROCESO PRINCIPAL
# ==============================================================================

def main():
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    maestro, adicional, renta = leer_fuentes()
    df = integrar(maestro, adicional, renta)
    df = reconstruir_precio(df)
    df = calcular_iav(df)
    df = crear_derivadas(df)

    # Series anuales de publicación tardía.
    series_tardias = [
        "renta_neta_media_por_hogar_eur",
        "renta_mediana_por_unidad_consumo_eur",
        "tamano_medio_hogar",
        "pct_hogares_unipersonales",
        "pct_poblacion_espanola",
        "pct_18_34", "pct_35_64", "pct_65_mas",
    ]
    df = arrastrar_series_anuales(df, series_tardias)

    df = df[list(VARIABLES_FINALES)].sort_values(
        ["provincia", "periodo"]
    ).reset_index(drop=True)

    # Redondeo homogéneo a tres decimales.
    numericas = df.select_dtypes(include=np.number).columns
    df[numericas] = df[numericas].round(DECIMALES)

    auditoria = construir_auditoria(df)

    df.to_csv(DIR_SALIDA / "dataset_TFM_MRBA.csv",
              sep=";", decimal=",", index=False, encoding="utf-8-sig")
    auditoria.to_csv(DIR_SALIDA / "auditoria_dataset_TFM_MRBA.csv",
                     sep=";", index=False, encoding="utf-8-sig")

    # ------------------------------ Resumen -----------------------------------
    print(f"Dataset generado: {df.shape[0]:,} observaciones × {df.shape[1]} variables")
    print(f"Territorios: {df['provincia'].nunique()} | Periodos: {df['periodo'].nunique()}")
    print(f"Predictoras: {(auditoria['Uso'] == 'Predictora').sum()}")
    print(f"Valores ausentes: {int(df.isna().sum().sum())}")
    print(f"Observaciones con series arrastradas: {int(df['valores_arrastrados'].sum())}")

    evolucion = df.groupby(df["periodo"].str[:4])["IAV_total"].mean()
    variacion = (evolucion.iloc[-1] / evolucion.iloc[0] - 1) * 100
    print(f"IAV medio: {evolucion.iloc[0]:.3f} (2015) -> "
          f"{evolucion.iloc[-1]:.3f} (2024) | {variacion:+.2f} %")

    return df, auditoria


if __name__ == "__main__":
    main()
