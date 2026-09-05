# -*- coding: utf-8 -*-
"""
================================================================================
TFM - Predicción de la accesibilidad a la vivienda en España

Construcción del conjunto de datos maestro provincial trimestral (2015-2024)

Autor: Marcos Rodrigo Bermejo Arroyo
================================================================================
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

ANIO_INICIO = 2015
ANIO_FIN = 2024

SUBCARPETAS = {
    "demografia": "Demografia",
    "economia": "Economía",
    "mercado_inmobiliario": "Mercado inmobiliario",
    "mercado_laboral": "Mercado laboral",
    "territorial": "Territorial",
    "turismo": "Turismo",
    "vivienda": "Vivienda",
}

# Las fuentes se leen directamente del repositorio del proyecto en GitHub.
# Para trabajar sin conexión basta con descargar la carpeta "fuentes" y
# apuntar URL_FUENTES a la ruta local correspondiente.
from urllib.parse import quote

URL_FUENTES = (
    "https://raw.githubusercontent.com/rodridemarcos/TFM_MRBA/"
    "refs/heads/main/fuentes"
)


def normalizar_texto(texto: object) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()



PROVINCIA_ALIAS = {
    "asturias principado de": "asturias",
    "las palmas": "palmas las",
    "s c tenerife": "santa cruz de tenerife",
    "alicante": "alicante alacant",
    "castellon": "castellon castello",
    "valencia": "valencia valencia",
    "a coruna": "coruna a",
    "madrid comunidad de": "madrid",
    "murcia region de": "murcia",
    "alava": "araba alava",
    "araba alava": "araba alava",
    "vizcaya": "bizkaia",
    "guipuzcoa": "gipuzkoa",
    "baleares": "balears illes",
    "navarra comunidad foral de": "navarra",
    "rioja la": "rioja la",
}


def normalizar_provincia_clave(texto: object) -> str:
    clave = normalizar_texto(texto)
    return PROVINCIA_ALIAS.get(clave, clave)


def leer_csv(raiz: Path, bloque: str, nombre: str,) -> pd.DataFrame:
    """Lee un fichero de fuente directamente desde el repositorio de GitHub.

    Se construye la URL a partir de la carpeta temática y el nombre del
    fichero, codificando los caracteres no ASCII de las subcarpetas
    (por ejemplo, la tilde de "Economía").
    """
    carpeta = quote(SUBCARPETAS[bloque])
    url = f"{URL_FUENTES}/{carpeta}/{nombre}"

    # Detecta automáticamente si el CSV utiliza coma o punto y coma
    # leyendo únicamente la primera línea.
    import urllib.request

    with urllib.request.urlopen(url) as respuesta:
        primera_linea = (
            respuesta.readline().decode("utf-8-sig", errors="replace")
        )

    separador = (
        ";"
        if primera_linea.count(";") > primera_linea.count(",")
        else ","
    )

    df = pd.read_csv(
        url,
        sep=separador,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print(
        f"Leído: {nombre:<65} "
        f"{df.shape} | separador='{separador}'"
    )

    return df


def preparar_claves_provincia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "codigo_provincia" in df.columns:
        df["codigo_provincia"] = pd.to_numeric(
            df["codigo_provincia"], errors="coerce"
        ).astype("Int64")

    if "provincia" in df.columns:
        df["provincia"] = df["provincia"].astype("string").str.strip()
        df["provincia_normalizada"] = df["provincia"].map(normalizar_provincia_clave)

    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")

    if "trimestre" in df.columns:
        df["trimestre"] = pd.to_numeric(
            df["trimestre"], errors="coerce"
        ).astype("Int64")

    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")

    return df


def merge_seguro(
    base: pd.DataFrame,
    derecha: pd.DataFrame,
    claves: list[str],
    nombre_bloque: str,
    incidencias: list[dict],
) -> pd.DataFrame:
    duplicados = int(derecha.duplicated(claves).sum())
    incidencias.append(
        {
            "bloque": nombre_bloque,
            "tipo": "duplicados_clave_fuente",
            "valor": duplicados,
        }
    )
    if duplicados:
        raise ValueError(
            f"{nombre_bloque}: hay {duplicados} duplicados en las claves {claves}."
        )

    n_antes = len(base)
    resultado = base.merge(derecha, on=claves, how="left", validate="many_to_one")
    if len(resultado) != n_antes:
        raise AssertionError(
            f"{nombre_bloque}: el merge alteró el número de filas."
        )

    columnas_datos = [c for c in derecha.columns if c not in claves]
    filas_con_dato = (
        int(resultado[columnas_datos].notna().any(axis=1).sum())
        if columnas_datos
        else 0
    )
    incidencias.append(
        {
            "bloque": nombre_bloque,
            "tipo": "filas_panel_con_algun_dato",
            "valor": filas_con_dato,
        }
    )
    return resultado


def construir_panel_base(
    actividad: pd.DataFrame,
    mapa_ccaa: pd.DataFrame,
) -> pd.DataFrame:
    columnas = [
        "codigo_provincia",
        "provincia",
        "anio",
        "trimestre",
        "tasa_actividad_pct_hombres",
        "tasa_empleo_pct_hombres",
        "tasa_paro_pct_hombres",
        "tasa_actividad_pct_mujeres",
        "tasa_empleo_pct_mujeres",
        "tasa_paro_pct_mujeres",
        "tasa_actividad_pct_total",
        "tasa_empleo_pct_total",
        "tasa_paro_pct_total",
    ]
    panel = actividad[columnas].copy()
    panel = panel.query("@ANIO_INICIO <= anio <= @ANIO_FIN").copy()

    panel["mes"] = panel["trimestre"] * 3
    panel["fecha"] = pd.PeriodIndex(
        year=panel["anio"].astype(int),
        quarter=panel["trimestre"].astype(int),
        freq="Q",
    ).to_timestamp(how="end").normalize()
    panel["periodo"] = (
        panel["anio"].astype(str)
        + "T"
        + panel["trimestre"].astype(str)
    )

    panel = panel.merge(
        mapa_ccaa,
        on="codigo_provincia",
        how="left",
        validate="many_to_one",
        suffixes=("", "_mapa"),
    )

    panel["brecha_actividad_genero_pp"] = (
        panel["tasa_actividad_pct_hombres"]
        - panel["tasa_actividad_pct_mujeres"]
    )
    panel["brecha_empleo_genero_pp"] = (
        panel["tasa_empleo_pct_hombres"]
        - panel["tasa_empleo_pct_mujeres"]
    )
    panel["brecha_paro_genero_pp"] = (
        panel["tasa_paro_pct_mujeres"]
        - panel["tasa_paro_pct_hombres"]
    )

    panel = panel.sort_values(
        ["codigo_provincia", "anio", "trimestre"]
    ).reset_index(drop=True)

    esperadas = 52 * (ANIO_FIN - ANIO_INICIO + 1) * 4
    if len(panel) != esperadas:
        raise AssertionError(
            f"El panel base tiene {len(panel)} filas y se esperaban {esperadas}."
        )
    return panel


def agregar_turismo_trimestral(
    demanda: pd.DataFrame,
    oferta: pd.DataFrame,
) -> pd.DataFrame:
    demanda = preparar_claves_provincia(demanda)
    oferta = preparar_claves_provincia(oferta)

    demanda["trimestre"] = ((demanda["mes"] - 1) // 3 + 1).astype("Int64")
    oferta["trimestre"] = ((oferta["mes"] - 1) // 3 + 1).astype("Int64")

    demanda_q = (
        demanda.groupby(
            ["codigo_provincia", "anio", "trimestre"],
            as_index=False,
        )
        .agg(
            viajeros_total=("viajeros_total", "sum"),
            viajeros_residentes_espana=("viajeros_residentes_espana", "sum"),
            viajeros_residentes_extranjero=("viajeros_residentes_extranjero", "sum"),
            pernoctaciones_total=("pernoctaciones_total", "sum"),
            pernoctaciones_residentes_espana=("pernoctaciones_residentes_espana", "sum"),
            pernoctaciones_residentes_extranjero=("pernoctaciones_residentes_extranjero", "sum"),
        )
    )
    demanda_q["estancia_media_total"] = np.where(
        demanda_q["viajeros_total"] > 0,
        demanda_q["pernoctaciones_total"] / demanda_q["viajeros_total"],
        np.nan,
    )

    oferta_q = (
        oferta.groupby(
            ["codigo_provincia", "anio", "trimestre"],
            as_index=False,
        )
        .agg(
            establecimientos_abiertos_estimados=("establecimientos_abiertos_estimados", "mean"),
            habitaciones_estimadas=("habitaciones_estimadas", "mean"),
            plazas_estimadas=("plazas_estimadas", "mean"),
            personal_empleado_hoteles=("personal_empleado", "mean"),
            grado_ocupacion_habitaciones_pct=("grado_ocupacion_habitaciones_pct", "mean"),
            grado_ocupacion_plazas_pct=("grado_ocupacion_plazas_pct", "mean"),
            grado_ocupacion_plazas_fin_semana_pct=("grado_ocupacion_plazas_fin_semana_pct", "mean"),
        )
    )
    return demanda_q.merge(
        oferta_q,
        on=["codigo_provincia", "anio", "trimestre"],
        how="outer",
        validate="one_to_one",
    )


def preparar_demografia(
    estructura: pd.DataFrame,
    indicadores: pd.DataFrame,
    poblacion_reciente: pd.DataFrame,
) -> pd.DataFrame:
    estructura = preparar_claves_provincia(estructura)
    indicadores = preparar_claves_provincia(indicadores)
    poblacion_reciente = preparar_claves_provincia(poblacion_reciente)

    estructura_sel = estructura[
        [
            "codigo_provincia",
            "anio",
            "poblacion_total",
            "edad_media_aprox",
            "pct_0_17",
            "pct_18_34",
            "pct_65_mas",
            "indice_envejecimiento",
            "tasa_dependencia",
            "pct_mujeres",
            "razon_masculinidad",
        ]
    ].copy()

    indicadores_sel = indicadores[
        [
            "provincia_normalizada",
            "anio",
            "tamano_medio_hogar",
            "pct_hogares_unipersonales",
            "pct_poblacion_espanola",
        ]
    ].copy()

    mapa_codigo = pd.concat(
        [
            estructura[["codigo_provincia", "provincia_normalizada"]],
            poblacion_reciente[["codigo_provincia", "provincia_normalizada"]],
        ],
        ignore_index=True,
    ).drop_duplicates("provincia_normalizada")

    indicadores_sel = indicadores_sel.merge(
        mapa_codigo,
        on="provincia_normalizada",
        how="left",
        validate="many_to_one",
    ).drop(columns="provincia_normalizada")

    demografia = estructura_sel.merge(
        indicadores_sel,
        on=["codigo_provincia", "anio"],
        how="outer",
        validate="one_to_one",
    )

    reciente_sel = poblacion_reciente[
        [
            "codigo_provincia",
            "anio",
            "poblacion_total",
            "pct_mujeres",
            "razon_masculinidad",
        ]
    ].copy()

    demografia = demografia.merge(
        reciente_sel,
        on=["codigo_provincia", "anio"],
        how="outer",
        suffixes=("", "_reciente"),
        validate="one_to_one",
    )
    for col in ["poblacion_total", "pct_mujeres", "razon_masculinidad"]:
        demografia[col] = demografia[f"{col}_reciente"].combine_first(
            demografia[col]
        )
        demografia = demografia.drop(columns=f"{col}_reciente")

    return demografia


def preparar_hogares_trimestral(hogares: pd.DataFrame) -> pd.DataFrame:
    hogares = preparar_claves_provincia(hogares)
    hogares["tamano_hogar"] = hogares["tamano_hogar"].astype(str).str.strip()

    hogares = hogares[
        hogares["tamano_hogar"].map(normalizar_texto) != "total"
    ].copy()

    pivot = hogares.pivot_table(
        index=["codigo_provincia", "anio", "trimestre"],
        columns="tamano_hogar",
        values="numero_hogares",
        aggfunc="sum",
    ).reset_index()

    renombrar = {
        "1": "hogares_1_persona_miles",
        "2": "hogares_2_personas_miles",
        "3": "hogares_3_personas_miles",
        "4": "hogares_4_personas_miles",
        "5 o más": "hogares_5_mas_personas_miles",
        "5 o mas": "hogares_5_mas_personas_miles",
    }
    pivot = pivot.rename(columns=renombrar)

    cols_hogar = [c for c in pivot.columns if c.startswith("hogares_")]
    pivot["hogares_total_miles"] = pivot[cols_hogar].sum(axis=1, min_count=1)

    if "hogares_1_persona_miles" in pivot.columns:
        pivot["pct_hogares_1_persona"] = (
            pivot["hogares_1_persona_miles"]
            / pivot["hogares_total_miles"]
            * 100
        )

    pesos = {
        "hogares_1_persona_miles": 1,
        "hogares_2_personas_miles": 2,
        "hogares_3_personas_miles": 3,
        "hogares_4_personas_miles": 4,
        "hogares_5_mas_personas_miles": 5,
    }
    numerador = sum(
        pivot[col] * peso
        for col, peso in pesos.items()
        if col in pivot.columns
    )
    pivot["tamano_medio_hogar_estimado"] = (
        numerador / pivot["hogares_total_miles"]
    )
    return pivot


def preparar_renta(
    renta: pd.DataFrame,
    mapa_nombre_codigo: pd.DataFrame,
) -> pd.DataFrame:
    renta = preparar_claves_provincia(renta)
    renta = renta.merge(
        mapa_nombre_codigo,
        on="provincia_normalizada",
        how="left",
        validate="many_to_one",
    )
    columnas = [
        "codigo_provincia",
        "anio",
        "renta_bruta_media_por_hogar_eur",
        "renta_bruta_media_por_persona_eur",
        "renta_mediana_por_unidad_consumo_eur",
        "renta_neta_media_por_hogar_eur",
        "renta_neta_media_por_persona_eur",
    ]
    return renta[columnas]


def preparar_salarios(
    salarios: pd.DataFrame,
    mapa_nombre_codigo: pd.DataFrame,
) -> pd.DataFrame:

    salarios = preparar_claves_provincia(salarios)

    # El nuevo fichero ya trae el código de provincia.
    if "codigo_provincia" not in salarios.columns:
        salarios = salarios.merge(
            mapa_nombre_codigo,
            on="provincia_normalizada",
            how="left",
            validate="many_to_one",
        )

    salarios = salarios.rename(
        columns={
            "salario_nac": "salario_medio_anual_euros_total",
            "salario_hombre": "salario_medio_anual_euros_hombre",
            "salario_mujer": "salario_medio_anual_euros_mujer",
        }
    )

    columnas = [
        "codigo_provincia",
        "anio",
        "salario_medio_anual_euros_total",
        "salario_medio_anual_euros_hombre",
        "salario_medio_anual_euros_mujer",
        "brecha_salarial_genero_pct",
    ]

    return salarios[columnas]


def preparar_alquiler_territorial(
    territorial: pd.DataFrame,
) -> pd.DataFrame:
    territorial = preparar_claves_provincia(territorial)
    columnas = [
        "codigo_provincia",
        "anio",
        "alquiler_mensual_eur_m2_mediana_vivienda_colectiva",
        "alquiler_mensual_total_eur_mediana_vivienda_colectiva",
        "superficie_m2_mediana_vivienda_colectiva",
        "observaciones_alquiler_vivienda_colectiva",
    ]
    return territorial[columnas]


def preparar_ipv_ccaa(
    ipv: pd.DataFrame,
    mapa_ccaa: pd.DataFrame,
) -> pd.DataFrame:
    ipv = ipv.copy()
    ipv["ccaa_normalizada"] = ipv["ccaa"].map(normalizar_texto)
    mapa = mapa_ccaa[
        ["codigo_ccaa", "comunidad_autonoma"]
    ].drop_duplicates("codigo_ccaa")
    mapa["ccaa_normalizada"] = mapa["comunidad_autonoma"].map(normalizar_texto)

    ipv = ipv.merge(
        mapa[["codigo_ccaa", "ccaa_normalizada"]],
        on="codigo_ccaa",
        how="left",
        suffixes=("", "_mapa"),
    )
    ipv["codigo_ccaa"] = pd.to_numeric(ipv["codigo_ccaa"], errors="coerce").astype("Int64")
    ipv["anio"] = pd.to_numeric(ipv["anio"], errors="coerce").astype("Int64")
    ipv["trimestre"] = pd.to_numeric(ipv["trimestre"], errors="coerce").astype("Int64")

    return ipv[
        [
            "codigo_ccaa",
            "anio",
            "trimestre",
            "ipv_general",
            "ipv_vivienda_nueva",
            "ipv_segunda_mano",
        ]
    ].drop_duplicates(["codigo_ccaa", "anio", "trimestre"])


def preparar_ipva_provincial(ipva: pd.DataFrame) -> pd.DataFrame:
    ipva = preparar_claves_provincia(ipva)
    return ipva[
        [
            "codigo_provincia",
            "anio",
            "ipva_indice_general",
            "ipva_variacion_anual_general",
        ]
    ]


def preparar_valor_tasado_trimestral(df: pd.DataFrame) -> pd.DataFrame:
    df = preparar_claves_provincia(df)
    # La columna calculada original contiene errores de escala en algunos registros.
    # Se reconstruye desde valor_original, que conserva el valor correcto.
    valor = (
        df["valor_original"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    valor_num = pd.to_numeric(valor, errors="coerce")

    # Cuando valor_original ya usa punto decimal, la conversión anterior lo elimina.
    valor_directo = pd.to_numeric(df["valor_original"], errors="coerce")
    usar_directo = valor_directo.between(300, 10000)
    df["valor_tasado_m2"] = np.where(
        usar_directo,
        valor_directo,
        valor_num,
    )

    return df[
        [
            "codigo_provincia",
            "anio",
            "trimestre",
            "valor_tasado_m2",
        ]
    ]


def preparar_suelo(
    df: pd.DataFrame,
    nombre_variable: str,
    mapa_nombre_codigo: pd.DataFrame,
) -> pd.DataFrame:
    df = preparar_claves_provincia(df)
    df = df.merge(
        mapa_nombre_codigo,
        on="provincia_normalizada",
        how="left",
        validate="many_to_one",
    )
    return df[
        ["codigo_provincia", "anio", "trimestre", nombre_variable]
    ]


def preparar_indicadores_nacionales_trimestrales(
    indicadores: pd.DataFrame,
) -> pd.DataFrame:
    indicadores = indicadores.copy()
    indicadores["fecha"] = pd.to_datetime(indicadores["fecha"], errors="coerce")
    indicadores["anio"] = indicadores["fecha"].dt.year
    indicadores["trimestre"] = indicadores["fecha"].dt.quarter

    seleccion = {
        "SI_1_5.1": "ipv_nacional_variacion_interanual_pct",
        "SI_1_5.13": "ipc_alquiler_variacion_interanual_pct",
        "SI_1_5.14": "indice_costes_construccion",
        "SI_1_5.31": "compraventas_viviendas_total",
        "SI_1_5.32": "compraventas_viviendas_nuevas",
        "SI_1_5.33": "compraventas_viviendas_usadas",
        "SI_1_5.37": "plazo_medio_hipotecas_meses",
        "SI_1_5.38": "ratio_prestamo_valor_pct",
        "SI_1_5.39": "prestamos_ltv_superior_80_pct",
        "SI_1_5.40": "tipo_interes_prestamos_vivienda_pct",
        "SI_1_5.42": "tasa_paro_joven_20_29_nacional_pct",
        "SI_1_5.43": "ratio_precio_vivienda_renta_hogar_anios_nacional",
        "SI_1_5.44": "esfuerzo_teorico_vivienda_pct_nacional",
        "SI_1_5.55": "credito_vivienda_variacion_interanual_pct",
        "SI_1_5.62": "rentabilidad_anual_vivienda_pct",
        "SI_1_5.63": "rentabilidad_bruta_alquiler_pct",
        "SI_1_5.82": "parque_viviendas_estimado_nacional",
        "SI_1_5.90": "pct_vivienda_principal_propiedad_nacional",
        "SI_1_5.91": "pct_vivienda_principal_alquiler_nacional",
    }

    datos = indicadores[
        indicadores["alias_serie"].isin(seleccion)
        & indicadores["anio"].between(ANIO_INICIO, ANIO_FIN)
    ].copy()

    # Para series mensuales, trimestrales y anuales se usa la media del trimestre.
    # En compraventas mensuales se usa la suma trimestral.
    alias_suma = {"SI_1_5.31", "SI_1_5.32", "SI_1_5.33"}
    partes = []
    for alias, nombre in seleccion.items():
        sub = datos.loc[datos["alias_serie"] == alias]
        if sub.empty:
            continue
        if (sub["frecuencia"] == "ANUAL").all():
            anual = sub.groupby("anio", as_index=False)["valor"].mean()
            trimestres = pd.DataFrame({"trimestre": [1, 2, 3, 4]})
            anual["_clave"] = 1
            trimestres["_clave"] = 1
            agg = anual.merge(trimestres, on="_clave").drop(columns="_clave")
            agg = agg[["anio", "trimestre", "valor"]]
        elif alias in alias_suma:
            agg = sub.groupby(["anio", "trimestre"], as_index=False)["valor"].sum()
        else:
            agg = sub.groupby(["anio", "trimestre"], as_index=False)["valor"].mean()
        agg = agg.rename(columns={"valor": nombre})
        partes.append(agg)

    resultado = None
    for parte in partes:
        resultado = (
            parte
            if resultado is None
            else resultado.merge(
                parte,
                on=["anio", "trimestre"],
                how="outer",
                validate="one_to_one",
            )
        )
    return resultado if resultado is not None else pd.DataFrame(
        columns=["anio", "trimestre"]
    )


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["viajeros_por_1000_habitantes"] = (
        df["viajeros_total"] / df["poblacion_total"] * 1000
    )
    df["pernoctaciones_por_habitante"] = (
        df["pernoctaciones_total"] / df["poblacion_total"]
    )
    df["plazas_hoteleras_por_1000_habitantes"] = (
        df["plazas_estimadas"] / df["poblacion_total"] * 1000
    )
    df["transacciones_suelo_por_100000_habitantes"] = (
        df["numero_transacciones"] / df["poblacion_total"] * 100000
    )

    df["precio_vivienda_estimado_eur"] = (
        df["valor_tasado_m2"]
        * df["superficie_m2_mediana_vivienda_colectiva"]
    )
    # =====================================================
    # Índice Provincial de Accesibilidad a la Vivienda (IAV)
    # =====================================================
    # Proporción del salario medio anual necesaria para adquirir
    # un metro cuadrado de vivienda en la provincia.
    #
    # Interpretación:
    #   - Un valor menor implica mayor accesibilidad.
    #   - Un valor mayor implica menor accesibilidad.
    df["IAV_total"] = np.where(
        df["salario_medio_anual_euros_total"] > 0,
        (
            df["valor_tasado_m2"]
            / df["salario_medio_anual_euros_total"]
        ),
        np.nan,
    )

    df["IAV_hombre"] = np.where(
        df["salario_medio_anual_euros_hombre"] > 0,
        (
            df["valor_tasado_m2"]
            / df["salario_medio_anual_euros_hombre"]
        ),
        np.nan,
    )

    df["IAV_mujer"] = np.where(
        df["salario_medio_anual_euros_mujer"] > 0,
        (
            df["valor_tasado_m2"]
            / df["salario_medio_anual_euros_mujer"]
        ),
        np.nan,
    )

    df["anios_renta_neta_hogar_para_compra"] = (
        df["precio_vivienda_estimado_eur"]
        / df["renta_neta_media_por_hogar_eur"]
    )
    df["anios_renta_bruta_hogar_para_compra"] = (
        df["precio_vivienda_estimado_eur"]
        / df["renta_bruta_media_por_hogar_eur"]
    )
    df["esfuerzo_alquiler_renta_neta_pct"] = (
        df["alquiler_mensual_total_eur_mediana_vivienda_colectiva"]
        * 12
        / df["renta_neta_media_por_hogar_eur"]
        * 100
    )
    df["alquiler_eur_m2_sobre_valor_tasado_m2_pct"] = (
        df["alquiler_mensual_eur_m2_mediana_vivienda_colectiva"]
        * 12
        / df["valor_tasado_m2"]
        * 100
    )

    df["valor_medio_transaccion_suelo_miles_eur"] = np.where(
        df["numero_transacciones"] > 0,
        df["valor_transacciones_miles_euros"]
        / df["numero_transacciones"],
        np.nan,
    )
    df["valor_suelo_estimado_miles_eur"] = (
        df["precio_suelo_m2"] * df["superficie_m2"] / 1000
    )

    # Dinámica temporal provincial sin usar información futura.
    df = df.sort_values(
        ["codigo_provincia", "anio", "trimestre"]
    ).reset_index(drop=True)
    grupo = df.groupby("codigo_provincia", group_keys=False)

    for variable in [
        "valor_tasado_m2",
        "precio_suelo_m2",
        "viajeros_total",
        "pernoctaciones_total",
        "tasa_paro_pct_total",
    ]:
        df[f"{variable}_variacion_interanual_pct"] = (
            grupo[variable].pct_change(4, fill_method=None) * 100
        )

    df["valor_tasado_m2_lag_1t"] = grupo["valor_tasado_m2"].shift(1)
    df["valor_tasado_m2_lag_4t"] = grupo["valor_tasado_m2"].shift(4)
    df["valor_tasado_m2_media_movil_4t"] = (
        grupo["valor_tasado_m2"]
        .rolling(4, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

    columnas_division = [
        c for c in df.columns
        if (
            "por_" in c
            or "esfuerzo_" in c
            or "anios_renta" in c
            or "valor_medio_" in c
            or c.startswith("IAV_")
        )
    ]
    df[columnas_division] = df[columnas_division].replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def construir_diccionario(
    df: pd.DataFrame,
    origen_columnas: dict[str, str],
) -> pd.DataFrame:
    filas = []
    for columna in df.columns:
        if columna in {
            "codigo_ccaa",
            "comunidad_autonoma",
            "codigo_provincia",
            "provincia",
            "fecha",
            "anio",
            "trimestre",
            "mes",
            "periodo",
        }:
            rol = "identificacion"
        elif (
            columna.startswith(("valor_tasado", "ipva_", "ipv_"))
            or columna.startswith("IAV_")
        ):
            rol = "objetivo_o_mercado_vivienda"
        elif any(
            token in columna
            for token in [
                "esfuerzo",
                "anios_renta",
                "por_1000",
                "por_100000",
                "por_habitante",
                "variacion_interanual",
                "lag_",
                "media_movil",
                "precio_vivienda_estimado",
                "valor_medio_transaccion",
                "valor_suelo_estimado",
                "brecha_",
            ]
        ):
            rol = "derivada"
        else:
            rol = "explicativa"

        filas.append(
            {
                "variable": columna,
                "tipo_dato": str(df[columna].dtype),
                "origen": origen_columnas.get(columna, "derivada"),
                "rol_propuesto": rol,
                "cobertura_pct": round(
                    df[columna].notna().mean() * 100, 2
                ),
            }
        )
    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye el panel maestro trimestral provincial 2015-2024 del TFM."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\UNIDAD USB\Máster\MÓDULOS\TFM\data\processed"),
        help="Carpeta raíz processed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Carpeta de salida. Por defecto: processed/Dataset conjunto.",
    )
    args = parser.parse_args()

    raiz = None
    salida = args.output or Path(__file__).parent
    salida.mkdir(parents=True, exist_ok=True)

    incidencias: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Lectura
    # ------------------------------------------------------------------
    actividad = preparar_claves_provincia(
        leer_csv(
            raiz,
            "mercado_laboral",
            "actividad_paro_empleo_provincia_trimestral.csv",
        )
    )
    poblacion_reciente = preparar_claves_provincia(
        leer_csv(
            raiz,
            "demografia",
            "poblacion_provincia_anio_2021_2025.csv",
        )
    )

    mapa_ccaa = (
        poblacion_reciente[
            [
                "codigo_ccaa",
                "comunidad_autonoma",
                "codigo_provincia",
                "provincia",
            ]
        ]
        .drop_duplicates("codigo_provincia")
        .copy()
    )
    mapa_nombre_codigo = mapa_ccaa[
        ["codigo_provincia", "provincia"]
    ].copy()
    mapa_nombre_codigo["provincia_normalizada"] = (
        mapa_nombre_codigo["provincia"].map(normalizar_provincia_clave)
    )
    mapa_nombre_codigo = mapa_nombre_codigo[
        ["codigo_provincia", "provincia_normalizada"]
    ].drop_duplicates("provincia_normalizada")

    panel = construir_panel_base(actividad, mapa_ccaa)

    origen_columnas = {
        c: "Mercado laboral"
        for c in panel.columns
        if c.startswith(("tasa_", "brecha_"))
    }
    origen_columnas.update(
        {
            "codigo_ccaa": "Demografía",
            "comunidad_autonoma": "Demografía",
            "codigo_provincia": "Mercado laboral",
            "provincia": "Mercado laboral",
            "fecha": "Construida",
            "anio": "Mercado laboral",
            "trimestre": "Mercado laboral",
            "mes": "Construida: último mes del trimestre",
            "periodo": "Construida",
        }
    )

    # ------------------------------------------------------------------
    # 2. Turismo mensual -> trimestral
    # ------------------------------------------------------------------
    turismo = agregar_turismo_trimestral(
        leer_csv(
            raiz,
            "turismo",
            "demanda_hotelera_provincia_mensual.csv",
        ),
        leer_csv(
            raiz,
            "turismo",
            "oferta_hotelera_provincia_mensual.csv",
        ),
    )
    panel = merge_seguro(
        panel,
        turismo,
        ["codigo_provincia", "anio", "trimestre"],
        "Turismo",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Turismo"
            for c in turismo.columns
            if c not in {"codigo_provincia", "anio", "trimestre"}
        }
    )

    # ------------------------------------------------------------------
    # 3. Demografía anual
    # ------------------------------------------------------------------
    demografia = preparar_demografia(
        leer_csv(
            raiz,
            "demografia",
            "estructura_demografica_provincia_anio_2015_2023.csv",
        ),
        leer_csv(
            raiz,
            "demografia",
            "indicadores_demograficos_provincia_anio_2015_2023.csv",
        ),
        poblacion_reciente,
    )
    panel = merge_seguro(
        panel,
        demografia,
        ["codigo_provincia", "anio"],
        "Demografía",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Demografía"
            for c in demografia.columns
            if c not in {"codigo_provincia", "anio"}
        }
    )

    hogares = preparar_hogares_trimestral(
        leer_csv(
            raiz,
            "vivienda",
            "hogares_tamano_trimestral.csv",
        )
    )
    panel = merge_seguro(
        panel,
        hogares,
        ["codigo_provincia", "anio", "trimestre"],
        "Hogares",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Vivienda - hogares"
            for c in hogares.columns
            if c not in {"codigo_provincia", "anio", "trimestre"}
        }
    )

    # ------------------------------------------------------------------
    # 4. Economía anual y salarios
    # ------------------------------------------------------------------
    renta = preparar_renta(
        leer_csv(raiz, "economia", "renta_provincia_anual.csv"),
        mapa_nombre_codigo,
    )
    panel = merge_seguro(
        panel,
        renta,
        ["codigo_provincia", "anio"],
        "Renta",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Economía - renta"
            for c in renta.columns
            if c not in {"codigo_provincia", "anio"}
        }
    )

    economia_nacional = leer_csv(
        raiz,
        "economia",
        "indicadores_economicos_nacionales_anual.csv",
    )
    economia_nacional["anio"] = pd.to_numeric(
        economia_nacional["anio"], errors="coerce"
    ).astype("Int64")
    panel = merge_seguro(
        panel,
        economia_nacional,
        ["anio"],
        "Economía nacional",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Economía nacional"
            for c in economia_nacional.columns
            if c != "anio"
        }
    )

    salarios = preparar_salarios(
        leer_csv(
            raiz,
            "mercado_laboral",
            "salarios_provinciales_2019_2024.csv",
        ),
        mapa_nombre_codigo,
    )
    panel = merge_seguro(
        panel,
        salarios,
        ["codigo_provincia", "anio"],
        "Salarios",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Mercado laboral - salarios"
            for c in salarios.columns
            if c not in {"codigo_provincia", "anio"}
        }
    )

    # ------------------------------------------------------------------
    # 5. Alquiler provincial histórico
    # ------------------------------------------------------------------
    alquiler = preparar_alquiler_territorial(
        leer_csv(
            raiz,
            "territorial",
            "territorial_provincia_anual.csv",
        )
    )
    panel = merge_seguro(
        panel,
        alquiler,
        ["codigo_provincia", "anio"],
        "Alquiler provincial",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Territorial - alquiler"
            for c in alquiler.columns
            if c not in {"codigo_provincia", "anio"}
        }
    )

    # ------------------------------------------------------------------
    # 6. Vivienda provincial y autonómica
    # ------------------------------------------------------------------
    ipv_ccaa = preparar_ipv_ccaa(
        leer_csv(raiz, "vivienda", "ipv_trimestral.csv"),
        mapa_ccaa,
    )
    panel = merge_seguro(
        panel,
        ipv_ccaa,
        ["codigo_ccaa", "anio", "trimestre"],
        "IPV por CCAA",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Vivienda - IPV"
            for c in ipv_ccaa.columns
            if c not in {"codigo_ccaa", "anio", "trimestre"}
        }
    )

    ipva = preparar_ipva_provincial(
        leer_csv(raiz, "vivienda", "ipva_provincial_anual.csv")
    )
    panel = merge_seguro(
        panel,
        ipva,
        ["codigo_provincia", "anio"],
        "IPVA provincial",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Vivienda - IPVA"
            for c in ipva.columns
            if c not in {"codigo_provincia", "anio"}
        }
    )

    valor_tasado = preparar_valor_tasado_trimestral(
        leer_csv(
            raiz,
            "vivienda",
            "valor_tasado_trimestral.csv",
        )
    )
    panel = merge_seguro(
        panel,
        valor_tasado,
        ["codigo_provincia", "anio", "trimestre"],
        "Valor tasado",
        incidencias,
    )
    origen_columnas["valor_tasado_m2"] = "Vivienda - valor tasado"

    fuentes_suelo = [
        ("suelo_trimestral.csv", "precio_suelo_m2"),
        ("superficie_suelo_trimestral.csv", "superficie_m2"),
        ("transacciones_suelo_trimestral.csv", "numero_transacciones"),
        (
            "valor_transacciones_suelo_trimestral.csv",
            "valor_transacciones_miles_euros",
        ),
    ]
    for archivo, variable in fuentes_suelo:
        suelo = preparar_suelo(
            leer_csv(raiz, "vivienda", archivo),
            variable,
            mapa_nombre_codigo,
        )
        panel = merge_seguro(
            panel,
            suelo,
            ["codigo_provincia", "anio", "trimestre"],
            variable,
            incidencias,
        )
        origen_columnas[variable] = "Vivienda - suelo"

    # ------------------------------------------------------------------
    # 7. Indicadores nacionales del mercado inmobiliario
    # ------------------------------------------------------------------
    mercado_nacional = preparar_indicadores_nacionales_trimestrales(
        leer_csv(
            raiz,
            "vivienda",
            "indicadores_mercado_inmobiliario_2015_2025.csv",
        )
    )
    panel = merge_seguro(
        panel,
        mercado_nacional,
        ["anio", "trimestre"],
        "Mercado inmobiliario nacional",
        incidencias,
    )
    origen_columnas.update(
        {
            c: "Mercado inmobiliario nacional"
            for c in mercado_nacional.columns
            if c not in {"anio", "trimestre"}
        }
    )

    # ------------------------------------------------------------------
    # 8. Variables derivadas
    # ------------------------------------------------------------------
    panel = crear_variables_derivadas(panel)

    origen_columnas.update(
        {
            "precio_vivienda_estimado_eur": (
                "Derivada: valor tasado por m² × superficie mediana"
            ),
            "IAV_total": (
                "Derivada: valor tasado por m² / salario medio anual total"
            ),
            "IAV_hombre": (
                "Derivada: valor tasado por m² / salario medio anual masculino"
            ),
            "IAV_mujer": (
                "Derivada: valor tasado por m² / salario medio anual femenino"
            ),
        }
    )

    # Se conservan solo columnas con algún dato en el periodo 2015-2024.
    columnas_vacias = [
        c for c in panel.columns
        if panel[c].isna().all()
    ]
    if columnas_vacias:
        panel = panel.drop(columns=columnas_vacias)

    # ------------------------------------------------------------------
    # 9. Validación
    # ------------------------------------------------------------------
    claves = ["codigo_provincia", "anio", "trimestre"]
    duplicados = int(panel.duplicated(claves).sum())
    if duplicados:
        raise AssertionError(
            f"El dataset final contiene {duplicados} claves duplicadas."
        )

    if panel["anio"].min() != ANIO_INICIO or panel["anio"].max() != ANIO_FIN:
        raise AssertionError("El periodo final no coincide con 2015-2024.")

    if panel["codigo_provincia"].nunique() != 52:
        raise AssertionError("El panel final no contiene 52 provincias.")

    for variable_iav in ["IAV_total", "IAV_hombre", "IAV_mujer"]:
        if variable_iav not in panel.columns:
            raise AssertionError(
                f"No se ha creado la variable objetivo {variable_iav}."
            )
        if np.isinf(panel[variable_iav].dropna()).any():
            raise AssertionError(
                f"La variable {variable_iav} contiene valores infinitos."
            )

    if panel["IAV_total"].notna().sum() == 0:
        raise AssertionError(
            "IAV_total no contiene observaciones válidas. "
            "Comprueba los salarios y el valor tasado."
        )

    auditoria = pd.DataFrame(
        [
            {"metrica": "filas_panel", "valor": len(panel)},
            {"metrica": "columnas_panel", "valor": panel.shape[1]},
            {"metrica": "provincias", "valor": panel["codigo_provincia"].nunique()},
            {"metrica": "ccaa", "valor": panel["codigo_ccaa"].nunique()},
            {"metrica": "anio_min", "valor": panel["anio"].min()},
            {"metrica": "anio_max", "valor": panel["anio"].max()},
            {"metrica": "trimestres", "valor": panel["periodo"].nunique()},
            {"metrica": "duplicados_clave", "valor": duplicados},
            {
                "metrica": "observaciones_IAV_total",
                "valor": int(panel["IAV_total"].notna().sum()),
            },
            {
                "metrica": "cobertura_IAV_total_pct",
                "valor": round(panel["IAV_total"].notna().mean() * 100, 2),
            },
            {
                "metrica": "observaciones_IAV_hombre",
                "valor": int(panel["IAV_hombre"].notna().sum()),
            },
            {
                "metrica": "observaciones_IAV_mujer",
                "valor": int(panel["IAV_mujer"].notna().sum()),
            },
            {
                "metrica": "porcentaje_celdas_no_nulas",
                "valor": round(panel.notna().mean().mean() * 100, 2),
            },
        ]
    )

    # ------------------------------------------------------------------
    # Auditoría del maestro: nombre, tipo, origen, rol y cobertura de cada
    # una de las 127 variables. Sustituye a los ficheros de diccionario y
    # cobertura que se generaban por separado.
    # ------------------------------------------------------------------
    auditoria_variables = construir_diccionario(panel, origen_columnas)

    # ------------------------------------------------------------------
    # Exportación: únicamente el conjunto maestro y su auditoría.
    # ------------------------------------------------------------------
    ruta_maestro = (
        salida / "dataset_maestro_vivienda_trimestral_2015_2024_IAV.csv"
    )
    ruta_auditoria = salida / "auditoria_dataset_maestro.csv"

    panel.to_csv(ruta_maestro, index=False, encoding="utf-8-sig")
    auditoria_variables.to_csv(
        ruta_auditoria, index=False, encoding="utf-8-sig"
    )

    print("\nDataset maestro generado correctamente")
    print("-" * 60)
    print(auditoria.to_string(index=False))
    print("\nArchivos exportados:")
    for ruta in (ruta_maestro, ruta_auditoria):
        print(f"  - {ruta}")


if __name__ == "__main__":
    main()