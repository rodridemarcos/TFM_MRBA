"""
================================================================================
APLICACIÓN DE ACCESIBILIDAD RESIDENCIAL — TFM
Predicción del Índice de Accesibilidad a la Vivienda (IAV) para 2025

Autor: Marcos Rodrigo Bermejo Arroyo
Máster en Big Data, Data Science & Artificial Intelligence

--------------------------------------------------------------------------------
EJECUCIÓN

    1. Instalar Flask:      pip install flask
    2. Colocar en la misma carpeta que este archivo:
         - modelo_TFM_MRBA.joblib
         - predicciones_2025.csv
         - matriz_prediccion_2025.csv
    3. Ejecutar este archivo en Spyder (F5)
    4. Abrir en el navegador: http://127.0.0.1:5000

--------------------------------------------------------------------------------
MÓDULOS

    1. Mapa      Cartograma de los 52 territorios con el IAV previsto para 2025
    2. Perfil    Cálculo personalizado a partir de los datos del usuario
    3. Escenario Reejecución del modelo modificando el tipo de interés

--------------------------------------------------------------------------------
ADVERTENCIA SOBRE EL ALCANCE DEL MODELO

El modelo predice el IAV PROVINCIAL, definido como los años de salario medio
de la provincia necesarios para adquirir una vivienda de 80 m². Los datos
personales que se introducen en el módulo 2 NO entran en el modelo: se aplican
aritméticamente sobre su salida, sustituyendo el salario provincial por el del
usuario y la superficie de referencia por la solicitada.

Los parámetros hipotecarios son SUPUESTOS DE LA APLICACIÓN y no resultados del
modelo. Se muestran de forma explícita y pueden modificarse.
================================================================================
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
import warnings

warnings.filterwarnings("ignore")
# ==============================================================================
# 1. PARÁMETROS Y SUPUESTOS
# ==============================================================================

DIRECTORIO = Path(__file__).parent

# Supuestos hipotecarios. NO proceden del modelo: son valores por defecto
# de la aplicación, editables por el usuario desde la interfaz.
SUPUESTOS = {
    "plazo_anios": 30,            # Plazo máximo habitual en España
    "plazo_maximo": 35,           # Excepcional, para compradores jóvenes
    "entrada_pct": 20.0,          # Porcentaje que no financia la entidad
    "gastos_compra_pct": 10.0,    # Impuestos y gastos de formalización
    "tipo_interes_pct": 3.2,      # Último dato del Banco de España
    "esfuerzo_maximo_pct": 30.0,  # Límite de endeudamiento sobre ingreso neto
    "esfuerzo_limite_pct": 35.0,  # Tolerancia máxima de las entidades
}

# ==============================================================================
# 2. CARTOGRAMA DE MOSAICO
# ==============================================================================

# Cada territorio ocupa una celda en una retícula que reproduce de forma
# esquemática la geografía peninsular. A diferencia de un mapa real, todas
# las provincias reciben la misma superficie, de modo que las de menor
# extensión resultan igual de legibles.

CARTOGRAMA = {
    # Disposición esquemática del territorio español. Cada provincia ocupa
    # una celda del mismo tamaño, de modo que las de menor extensión
    # resultan tan legibles como las grandes.
    "Coruña, A": (0, 0), "Lugo": (0, 1), "Asturias": (0, 2),
    "Cantabria": (0, 3), "Bizkaia": (0, 4), "Gipuzkoa": (0, 5),
    "Navarra": (0, 6), "Huesca": (0, 7), "Lleida": (0, 8),
    "Girona": (0, 9),

    "Pontevedra": (1, 0), "Ourense": (1, 1), "León": (1, 2),
    "Palencia": (1, 3), "Burgos": (1, 4), "Araba/Álava": (1, 5),
    "Rioja, La": (1, 6), "Zaragoza": (1, 7), "Barcelona": (1, 8),

    "Zamora": (2, 2), "Valladolid": (2, 3), "Soria": (2, 5),
    "Teruel": (2, 7), "Tarragona": (2, 8),

    "Salamanca": (3, 2), "Ávila": (3, 3), "Segovia": (3, 4),
    "Madrid": (3, 5), "Guadalajara": (3, 6),

    "Cáceres": (4, 2), "Toledo": (4, 5), "Cuenca": (4, 6),
    "Castellón/Castelló": (4, 7), "Balears, Illes": (4, 10),

    "Badajoz": (5, 2), "Ciudad Real": (5, 5), "Albacete": (5, 6),
    "Valencia/València": (5, 7),

    "Córdoba": (6, 4), "Jaén": (6, 5), "Murcia": (6, 6),
    "Alicante/Alacant": (6, 7),

    "Huelva": (7, 2), "Sevilla": (7, 3), "Granada": (7, 5),
    "Almería": (7, 6),

    "Cádiz": (8, 3), "Málaga": (8, 4),

    "Palmas, Las": (9, 0), "Santa Cruz de Tenerife": (9, 1),
    "Ceuta": (9, 3), "Melilla": (9, 4),
}

# Abreviaturas de tres caracteres para las etiquetas del mosaico.
ABREVIATURAS = {
    "Araba/Álava": "ARA", "Albacete": "ALB", "Alicante/Alacant": "ALI",
    "Almería": "ALM", "Asturias": "AST", "Ávila": "AVI", "Badajoz": "BAD",
    "Balears, Illes": "BAL", "Barcelona": "BCN", "Bizkaia": "BIZ",
    "Burgos": "BUR", "Cáceres": "CAC", "Cádiz": "CAD", "Cantabria": "CTB",
    "Castellón/Castelló": "CAS", "Ceuta": "CEU", "Ciudad Real": "CRE",
    "Córdoba": "COR", "Coruña, A": "ACO", "Cuenca": "CUE", "Girona": "GIR",
    "Granada": "GRA", "Guadalajara": "GUA", "Gipuzkoa": "GIP",
    "Huelva": "HUV", "Huesca": "HUE", "Jaén": "JAE", "León": "LEO",
    "Lleida": "LLE", "Lugo": "LUG", "Madrid": "MAD", "Málaga": "MAL",
    "Melilla": "MEL", "Murcia": "MUR", "Navarra": "NAV", "Ourense": "OUR",
    "Palencia": "PAL", "Palmas, Las": "LPA", "Pontevedra": "PON",
    "Rioja, La": "RIO", "Salamanca": "SAL",
    "Santa Cruz de Tenerife": "TFE", "Segovia": "SEG", "Sevilla": "SEV",
    "Soria": "SOR", "Tarragona": "TAR", "Teruel": "TER", "Toledo": "TOL",
    "Valencia/València": "VAL", "Valladolid": "VLL", "Zamora": "ZAM",
    "Zaragoza": "ZAR",
}


# ==============================================================================
# 3. CARGA DEL MODELO Y LOS DATOS
# ==============================================================================

def cargar_recursos():
    """Carga el modelo serializado y los conjuntos de datos asociados."""
    paquete = joblib.load(DIRECTORIO / "modelo_TFM_MRBA.joblib")

    predicciones = pd.read_csv(
        DIRECTORIO / "predicciones_2025.csv",
        sep=";", decimal=",", encoding="utf-8-sig",
    )

    matriz = pd.read_csv(
        DIRECTORIO / "matriz_prediccion_2025.csv",
        sep=";", decimal=",", encoding="utf-8-sig",
    )
    
    columnas_necesarias = set(paquete["columnas_matriz"]) | {
        "provincia", "persistencia"
    }
    faltan = columnas_necesarias - set(matriz.columns)

    if faltan:
        raise RuntimeError(
            "Faltan columnas en matriz_prediccion_2025.csv: "
            f"{sorted(faltan)}. Vuelva a ejecutar la celda de "
            "serialización del cuaderno."
        )

    return paquete, predicciones, matriz


PAQUETE, PREDICCIONES, MATRIZ = cargar_recursos()
COLUMNAS = PAQUETE["columnas_matriz"]
SUPERFICIE_REFERENCIA = PAQUETE["superficie_referencia_m2"]


# ==============================================================================
# 4. FUNCIONES DE CÁLCULO
# ==============================================================================

def predecir(matriz_entrada, persistencia):
    """Promedia las predicciones de los siete modelos.

    Devuelve además la dispersión entre ellos, que constituye una medida
    directa de la incertidumbre asociada a cada predicción.
    """
    salidas = []

    for contenido in PAQUETE["modelos"].values():
        prediccion = contenido["estimador"].predict(matriz_entrada)

        if contenido["formulacion"] == "Corrección":
            prediccion = persistencia + prediccion

        salidas.append(prediccion)

    salidas = np.vstack(salidas)

    return {
        "media": salidas.mean(axis=0),
        "minimo": salidas.min(axis=0),
        "maximo": salidas.max(axis=0),
        "dispersion": salidas.std(axis=0),
    }


def simular_escenario(variacion_tipo_interes):
    """Reejecuta el modelo modificando el tipo de interés hipotecario.

    El capítulo 5 identificó esta variable como determinante: su retirada
    duplicaba el error del modelo. El simulador permite comprobar cómo
    responde la predicción ante variaciones de la política monetaria.
    """
    matriz = MATRIZ.copy()
    columna_tipo = "tipo_interes_prestamos_vivienda_pct_lag1"

    matriz[columna_tipo] = matriz[columna_tipo] + variacion_tipo_interes

    entrada = matriz[COLUMNAS]
    persistencia = MATRIZ["persistencia"].values

    resultado = predecir(entrada, persistencia)

    salida = pd.DataFrame({
        "provincia": matriz["provincia"],
        "IAV": resultado["media"],
        "minimo": resultado["minimo"],
        "maximo": resultado["maximo"],
    })

    return salida.groupby("provincia", as_index=False).mean(numeric_only=True)


def cuota_hipotecaria(principal, tipo_anual, plazo_anios):
    """Cuota mensual constante de un préstamo de amortización francesa."""
    interes = tipo_anual / 100 / 12
    meses = plazo_anios * 12

    if meses <= 0:
        return 0.0
    if interes <= 0:
        return principal / meses

    factor = (1 + interes) ** meses
    return principal * interes * factor / (factor - 1)


def principal_maximo(cuota, tipo_anual, plazo_anios):
    """Importe máximo financiable dada una cuota mensual asumible."""
    interes = tipo_anual / 100 / 12
    meses = plazo_anios * 12

    if meses <= 0:
        return 0.0
    if interes <= 0:
        return cuota * meses

    factor = (1 + interes) ** meses
    return cuota * (factor - 1) / (interes * factor)


def calcular_perfil(datos):
    """Determina en qué territorios resulta accesible la vivienda buscada.

    El cálculo reproduce el criterio que aplican las entidades financieras.
    La capacidad de pago se obtiene restando las deudas vigentes al
    porcentaje máximo de endeudamiento sobre el ingreso neto, y de ella se
    deduce el importe máximo financiable. El precio asequible resulta del
    menor de dos límites: el que impone esa capacidad de pago y el que
    impone el ahorro disponible para la entrada y los gastos.

    Ninguno de estos valores interviene en el modelo predictivo. Se aplican
    sobre el precio que este anticipa para cada territorio.
    """
    salario_bruto = float(datos["salario_bruto"])
    superficie = float(datos["superficie"])
    sexo = datos.get("sexo", "total")
    con_hipoteca = datos.get("hipoteca", "si") == "si"

    gastos_pct = SUPUESTOS["gastos_compra_pct"]

    if con_hipoteca:
        ingreso_neto = float(datos.get("ingreso_neto", 0))
        deudas = float(datos.get("deudas", 0))
        ahorros = float(datos.get("ahorros", 0))
        plazo = float(datos.get("plazo", SUPUESTOS["plazo_anios"]))
        entrada_pct = float(datos.get("entrada", SUPUESTOS["entrada_pct"]))
        tipo = float(datos.get("tipo", SUPUESTOS["tipo_interes_pct"]))
        esfuerzo_max = float(
            datos.get("esfuerzo", SUPUESTOS["esfuerzo_maximo_pct"])
        )

        # Capacidad de pago mensual admitida por la entidad.
        capacidad = max(ingreso_neto * esfuerzo_max / 100 - deudas, 0)
        financiable = principal_maximo(capacidad, tipo, plazo)

        # Doble límite: capacidad de pago y ahorro disponible.
        limite_capacidad = (
            financiable / (1 - entrada_pct / 100)
            if entrada_pct < 100 else np.inf
        )
        limite_ahorro = ahorros / ((entrada_pct + gastos_pct) / 100)
        precio_maximo = min(limite_capacidad, limite_ahorro)
    else:
        presupuesto = float(datos.get("presupuesto", 0))
        ahorros = presupuesto
        ingreso_neto = deudas = capacidad = financiable = 0.0
        plazo = entrada_pct = tipo = esfuerzo_max = 0.0
        limite_capacidad = np.inf
        limite_ahorro = presupuesto / (1 + gastos_pct / 100)
        precio_maximo = limite_ahorro

    resumen = {
        "capacidad_mensual": round(capacidad, 0),
        "importe_financiable": round(financiable, 0),
        "limite_por_capacidad": (
            round(limite_capacidad, 0)
            if np.isfinite(limite_capacidad) else None
        ),
        "limite_por_ahorro": round(limite_ahorro, 0),
        "precio_maximo": round(precio_maximo, 0),
    }

    filas = []

    for _, provincia in PREDICCIONES.iterrows():
        precio_m2 = provincia["precio_m2_previsto"]
        coste = precio_m2 * superficie
        gastos = coste * gastos_pct / 100

        if sexo == "hombre":
            salario_referencia = provincia["salario_hombre_2024"]
        elif sexo == "mujer":
            salario_referencia = provincia["salario_mujer_2024"]
        else:
            salario_referencia = provincia["salario_2024"]

        iav_referencia = coste / salario_referencia
        iav_personal = (
            coste / salario_bruto if salario_bruto > 0 else np.inf
        )

        if con_hipoteca:
            entrada = coste * entrada_pct / 100
            desembolso_inicial = entrada + gastos
            prestamo = coste - entrada
            cuota = cuota_hipotecaria(prestamo, tipo, plazo)
            esfuerzo = (
                (cuota + deudas) / ingreso_neto * 100
                if ingreso_neto > 0 else np.inf
            )

            cubre_ahorro = ahorros >= desembolso_inicial
            admite_entidad = esfuerzo <= esfuerzo_max

            viable = cubre_ahorro and admite_entidad
            if not cubre_ahorro and not admite_entidad:
                motivo = "Ahorro y renta insuficientes"
            elif not cubre_ahorro:
                motivo = "Ahorro insuficiente"
            elif not admite_entidad:
                motivo = "Supera el límite de endeudamiento"
            else:
                motivo = "Viable"

            falta = max(desembolso_inicial - ahorros, 0)
        else:
            entrada = coste
            desembolso_inicial = coste + gastos
            prestamo = cuota = 0.0
            esfuerzo = 0.0
            viable = ahorros >= desembolso_inicial
            motivo = "Viable" if viable else "Presupuesto insuficiente"
            falta = max(desembolso_inicial - ahorros, 0)

        # Superficie alcanzable con el precio máximo asequible.
        superficie_asequible = (
            precio_maximo / precio_m2 if precio_m2 > 0 else 0.0
        )

        filas.append({
            "provincia": provincia["provincia"],
            "iav_provincial": round(provincia["IAV_previsto"], 2),
            "iav_referencia": round(iav_referencia, 2),
            "iav_personal": (
                round(iav_personal, 2) if np.isfinite(iav_personal) else None
            ),
            "precio_m2": round(precio_m2, 0),
            "coste": round(coste, 0),
            "gastos": round(gastos, 0),
            "entrada": round(entrada, 0),
            "desembolso_inicial": round(desembolso_inicial, 0),
            "falta_ahorro": round(falta, 0),
            "prestamo": round(prestamo, 0),
            "cuota": round(cuota, 0),
            "esfuerzo": round(esfuerzo, 1) if np.isfinite(esfuerzo) else None,
            "superficie_asequible": round(superficie_asequible, 0),
            "viable": bool(viable),
            "motivo": motivo,
        })

    return {"resumen": resumen, "territorios": filas}


# ==============================================================================
# 5. APLICACIÓN WEB
# ==============================================================================

app = Flask(__name__)


@app.route("/")
def inicio():
    return render_template_string(
        PLANTILLA,
        cartograma=CARTOGRAMA,
        abreviaturas=ABREVIATURAS,
        supuestos=SUPUESTOS,
        provincias=sorted(PREDICCIONES["provincia"].tolist()),
        metricas=PAQUETE["metricas_2024"],
    )


@app.route("/api/mapa")
def api_mapa():
    """Devuelve la predicción de 2025 para los 52 territorios."""
    datos = PREDICCIONES.copy()
    datos["celda"] = datos["provincia"].map(CARTOGRAMA)
    datos["abreviatura"] = datos["provincia"].map(ABREVIATURAS)

    return jsonify(
        datos[[
            "provincia", "comunidad_autonoma", "abreviatura",
            "IAV_previsto", "IAV_minimo", "IAV_maximo",
            "IAV_2024", "variacion_pct", "precio_m2_previsto",
            "salario_2024", "dispersion",
        ]].round(3).to_dict("records")
    )


@app.route("/api/perfil", methods=["POST"])
def api_perfil():
    return jsonify(calcular_perfil(request.json))


@app.route("/api/escenario", methods=["POST"])
def api_escenario():
    """Reejecuta el modelo con un tipo de interés modificado."""
    variacion = float(request.json.get("variacion", 0.0))

    resultado = simular_escenario(variacion)
    base = PREDICCIONES[["provincia", "IAV_previsto"]]

    comparacion = resultado.merge(base, on="provincia")
    comparacion["diferencia"] = (
        comparacion["IAV"] - comparacion["IAV_previsto"]
    )

    return jsonify({
        "detalle": comparacion.round(3).to_dict("records"),
        "media": round(comparacion["IAV"].mean(), 3),
        "media_base": round(comparacion["IAV_previsto"].mean(), 3),
        "variacion_media": round(comparacion["diferencia"].mean(), 3),
    })


# ==============================================================================
# 6. INTERFAZ
# ==============================================================================

PLANTILLA = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Accesibilidad residencial en España · Previsión 2025</title>
<style>
:root{
  --fondo:#0d1117; --panel:#161b22; --panel2:#1c2333; --borde:#2d3748;
  --texto:#e6edf3; --suave:#8b949e; --acento:#58a6ff; --acento2:#3fb950;
  --alerta:#f85149; --aviso:#d29922;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
  background:var(--fondo);
  background-image:radial-gradient(circle at 15% 10%,#1b2a4a 0%,transparent 45%),
                   radial-gradient(circle at 85% 90%,#132b25 0%,transparent 45%);
  background-attachment:fixed;
  color:var(--texto); line-height:1.55; padding:0 0 60px;
}
header{
  padding:28px 40px 22px; border-bottom:1px solid var(--borde);
  background:rgba(13,17,23,.82); backdrop-filter:blur(12px);
  position:sticky; top:0; z-index:50;
}
h1{font-size:21px;font-weight:600;letter-spacing:-.3px}
h1 span{color:var(--acento)}
.sub{color:var(--suave);font-size:13px;margin-top:4px}
nav{display:flex;gap:6px;margin-top:18px}
nav button{
  background:transparent;border:1px solid var(--borde);color:var(--suave);
  padding:9px 18px;border-radius:8px;cursor:pointer;font-size:13.5px;
  font-family:inherit;transition:.18s;
}
nav button:hover{border-color:var(--acento);color:var(--texto)}
nav button.activa{background:var(--acento);border-color:var(--acento);color:#06131f;font-weight:600}
main{max-width:1340px;margin:0 auto;padding:28px 40px}
.vista{display:none} .vista.activa{display:block}
.tarjeta{
  background:var(--panel);border:1px solid var(--borde);border-radius:14px;
  padding:24px;margin-bottom:20px;
}
.tarjeta h2{font-size:16px;font-weight:600;margin-bottom:6px}
.tarjeta p.desc{color:var(--suave);font-size:13px;margin-bottom:18px}
.rejilla{display:grid;grid-template-columns:1fr 330px;gap:20px}
@media(max-width:1050px){.rejilla{grid-template-columns:1fr}}
svg{width:100%;height:auto;display:block}
.celda{cursor:pointer;transition:.15s}
.celda:hover{stroke:#fff;stroke-width:2.5}
.etiqueta{font-size:11px;font-weight:700;fill:#fff;pointer-events:none;
  text-anchor:middle;text-shadow:0 1px 3px rgba(0,0,0,.85)}
.valor{font-size:9px;fill:rgba(255,255,255,.9);pointer-events:none;text-anchor:middle}
#detalle{position:sticky;top:150px}
.dato{display:flex;justify-content:space-between;padding:9px 0;
  border-bottom:1px solid var(--borde);font-size:13.5px}
.dato:last-child{border:none}
.dato span:first-child{color:var(--suave)}
.dato span:last-child{font-weight:600;font-variant-numeric:tabular-nums}
.grande{font-size:34px;font-weight:700;color:var(--acento);
  font-variant-numeric:tabular-nums;letter-spacing:-1px}
.leyenda{display:flex;align-items:center;gap:10px;margin-top:16px;font-size:12px;color:var(--suave)}
.barra{height:11px;flex:1;border-radius:6px;
  background:linear-gradient(90deg,#2c7fb8,#7fcdbb,#fed976,#fd8d3c,#e31a1c)}
form{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:15px}
label{display:block;font-size:12px;color:var(--suave);margin-bottom:5px;font-weight:500}
input,select{
  width:100%;background:var(--panel2);border:1px solid var(--borde);
  color:var(--texto);padding:9px 11px;border-radius:8px;font-size:13.5px;
  font-family:inherit;
}
input:focus,select:focus{outline:none;border-color:var(--acento)}
.boton{
  background:var(--acento);color:#06131f;border:none;padding:11px 26px;
  border-radius:8px;font-weight:600;cursor:pointer;font-size:14px;
  font-family:inherit;transition:.18s;
}
.boton:hover{filter:brightness(1.12)}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th{text-align:left;color:var(--suave);font-weight:500;padding:9px 8px;
  border-bottom:1px solid var(--borde);font-size:11.5px;text-transform:uppercase;
  letter-spacing:.5px}
td{padding:9px 8px;border-bottom:1px solid rgba(45,55,72,.4);
  font-variant-numeric:tabular-nums}
tr:hover td{background:rgba(88,166,255,.06)}
.viable{color:var(--acento2);font-weight:600}
.inviable{color:var(--alerta)}
.nota{
  background:rgba(210,153,34,.08);border-left:3px solid var(--aviso);
  padding:13px 16px;border-radius:0 8px 8px 0;font-size:12.5px;
  color:#d8c9a3;margin-bottom:18px;
}
.deslizador{display:flex;align-items:center;gap:16px;margin:20px 0}
input[type=range]{flex:1;accent-color:var(--acento)}
.metricas{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px}
.metrica{background:var(--panel2);border-radius:10px;padding:15px;text-align:center}
.metrica .n{font-size:23px;font-weight:700;color:var(--acento);
  font-variant-numeric:tabular-nums}
.metrica .t{font-size:11px;color:var(--suave);margin-top:3px}
.cargando{color:var(--suave);font-size:13px;padding:16px 0}
.mini{background:transparent;border:1px solid var(--borde);color:var(--suave);
  padding:6px 15px;border-radius:7px;cursor:pointer;font-size:12.5px;
  font-family:inherit;transition:.18s}
.mini:hover{border-color:var(--acento);color:var(--texto)}
.mini.activa{background:var(--acento);border-color:var(--acento);
  color:#06131f;font-weight:600}
.prov{cursor:pointer;transition:.15s;stroke:#0d1117;stroke-width:.7}
.prov:hover{stroke:#fff;stroke-width:2}
</style>
</head>
<body>

<header>
  <h1>Accesibilidad residencial en España · <span>Previsión 2025</span></h1>
  <div class="sub">
    Previsión elaborada combinando siete algoritmos de aprendizaje automático ·
    Al comprobarlo sobre 2024 se equivocó de media en 1,6 meses de salario,
    un 2% del valor real · La mitad de error que suponer que nada cambia
  </div>
  <nav>
    <button class="activa" data-vista="mapa">Mapa territorial</button>
    <button data-vista="perfil">Simulador de compra</button>
    <button data-vista="escenario">Efecto de los tipos de interés</button>
  </nav>
</header>

<main>

<!-- ============ MÓDULO 1: MAPA ============ -->
<section id="mapa" class="vista activa">
  <div class="rejilla">
    <div class="tarjeta">
      <h2>Índice de Accesibilidad a la Vivienda previsto para 2025</h2>
      <p class="desc">
        Años de salario bruto medio provincial necesarios para adquirir una
        vivienda de 80 m². Un valor de 6 significa que hacen falta seis años
        de sueldo íntegro, sin descontar ningún gasto. Pase el cursor sobre un
        territorio para consultarlo y pulse sobre él para fijarlo.
      </p>
      <div style="display:flex;gap:8px;margin-bottom:14px">
        <button class="mini activa" id="b-mapa">Mapa</button>
        <button class="mini" id="b-mosaico">Mosaico</button>
        <span id="aviso-mapa" style="color:var(--suave);font-size:12px;
              align-self:center;margin-left:8px"></span>
      </div>
      <svg id="mapageo" viewBox="0 0 1000 820"></svg>
      <svg id="carto" viewBox="0 0 650 593" style="display:none"></svg>
      <div class="leyenda">
        <span>Más accesible</span>
        <div class="barra"></div>
        <span>Menos accesible</span>
      </div>
    </div>

    <div class="tarjeta" id="detalle">
      <h2 id="d-nombre">Seleccione un territorio</h2>
      <p class="desc" id="d-ccaa">&nbsp;</p>
      <div class="grande" id="d-iav">—</div>
      <p class="desc">años de salario previstos para 2025</p>
      <div id="d-datos"></div>
    </div>
  </div>
</section>

<!-- ============ MÓDULO 2: PERFIL ============ -->
<section id="perfil" class="vista">
  <div class="tarjeta">
    <h2>Su situación financiera</h2>
    <p class="desc">
      El modelo predice el precio de la vivienda en cada territorio. Estos
      datos no intervienen en esa predicción: se aplican sobre ella para
      determinar dónde resulta accesible una vivienda con su situación
      financiera, siguiendo los criterios de concesión de las entidades.
    </p>
    <form id="f-perfil">
      <div>
        <label>Forma de compra</label>
        <select name="hipoteca" id="modo-compra">
          <option value="si">Con hipoteca</option>
          <option value="no">Al contado</option>
        </select>
      </div>
      <div>
        <label>Superficie buscada (m²)</label>
        <input type="number" name="superficie" value="80" min="20" max="400">
      </div>
      <div>
        <label>Salario bruto anual (€)</label>
        <input type="number" name="salario_bruto" value="28000" min="0" step="500">
      </div>
      <div class="solo-hipoteca">
        <label>Ingresos netos mensuales (€)</label>
        <input type="number" name="ingreso_neto" value="1800" min="0" step="50">
      </div>
      <div class="solo-hipoteca">
        <label>Cuotas mensuales de otras deudas (€)</label>
        <input type="number" name="deudas" value="0" min="0" step="25">
      </div>
      <div class="solo-hipoteca">
        <label>Ahorro disponible (€)</label>
        <input type="number" name="ahorros" value="45000" min="0" step="1000">
      </div>
      <div class="solo-contado" style="display:none">
        <label>Presupuesto disponible (€)</label>
        <input type="number" name="presupuesto" value="150000" min="0" step="5000">
      </div>
      <div>
        <label>Referencia salarial territorial</label>
        <select name="sexo">
          <option value="total">Media provincial</option>
          <option value="hombre">Salario masculino</option>
          <option value="mujer">Salario femenino</option>
        </select>
      </div>
    </form>

    <h2 style="margin-top:26px" class="solo-hipoteca">Supuestos hipotecarios</h2>
    <div class="nota solo-hipoteca">
      Estos valores reproducen los criterios habituales de concesión en España
      y son <strong>supuestos de la aplicación</strong>, no resultados del
      modelo. Las entidades exigen una entrada del 20% del valor del inmueble
      y no financian los gastos de compraventa, que rondan otro 10% en concepto
      de impuestos y formalización. El plazo máximo se sitúa en 30 años, con 35
      de forma excepcional para compradores jóvenes. La cuota, sumada al resto
      de deudas vigentes, no debe superar el 30% de los ingresos netos, con un
      35% como tolerancia máxima.
      <br><br>
      En <strong>obra nueva</strong> la entrada se abona de forma escalonada
      durante la construcción, mientras que en <strong>segunda mano</strong>
      debe disponerse del importe íntegro desde el primer día. La aplicación
      adopta el criterio más exigente, el de segunda mano.
    </div>
    <form id="f-supuestos" class="solo-hipoteca">
      <div>
        <label>Plazo (años, máximo 35)</label>
        <input type="number" name="plazo" value="{{ supuestos.plazo_anios }}"
               min="5" max="{{ supuestos.plazo_maximo }}">
      </div>
      <div>
        <label>Entrada (%)</label>
        <input type="number" name="entrada" value="{{ supuestos.entrada_pct }}"
               min="0" max="100" step="5">
      </div>
      <div>
        <label>Tipo de interés (%)</label>
        <input type="number" name="tipo" value="{{ supuestos.tipo_interes_pct }}"
               min="0" max="15" step="0.1">
      </div>
      <div>
        <label>Endeudamiento máximo (% del neto)</label>
        <input type="number" name="esfuerzo" value="{{ supuestos.esfuerzo_maximo_pct }}"
               min="10" max="{{ supuestos.esfuerzo_limite_pct }}" step="1">
      </div>
    </form>

    <div style="margin-top:22px">
      <button class="boton" id="b-calcular">Calcular</button>
    </div>
  </div>

  <div class="tarjeta">
    <h2>Resultado por territorio</h2>
    <p class="desc" id="p-resumen">Introduzca sus datos y pulse Calcular.</p>
    <div class="nota" id="nota-esfuerzo">
      El <strong>esfuerzo</strong> es la proporción de sus ingresos netos
      mensuales que absorberían la cuota hipotecaria y el resto de deudas
      vigentes. El <strong>desembolso inicial</strong> es la suma de la entrada
      que no financia la entidad y de los impuestos y gastos de formalización,
      importe que debe estar disponible en el momento de la compra. Los
      <strong>m² asequibles</strong> indican la superficie máxima que podría
      adquirir en ese territorio con su capacidad de compra.
    </div>
    <div style="max-height:520px;overflow-y:auto">
      <table id="t-perfil"></table>
    </div>
  </div>
</section>

<!-- ============ MÓDULO 3: ESCENARIO ============ -->
<section id="escenario" class="vista">
  <div class="tarjeta">
    <h2>Simulación de política monetaria</h2>
    <p class="desc">
      El tipo de interés de los préstamos para vivienda es la variable de la
      que más depende el modelo: al retirarla del conjunto, su error se
      duplica. Este simulador permite plantear la pregunta inversa, es decir,
      qué ocurriría con la accesibilidad si el Banco Central Europeo endureciera
      o relajara su política monetaria.
    </p>
    <div class="nota">
      El tipo vigente en el último dato disponible es del
      <strong>{{ supuestos.tipo_interes_pct }}%</strong>. Desplace el control
      para sumar o restar puntos porcentuales sobre ese valor. Al pulsar el
      botón, los siete modelos vuelven a ejecutarse con el tipo modificado y
      devuelven una previsión nueva para los 52 territorios: no se trata de
      resultados almacenados de antemano.
      <br><br>
      Un tipo <strong>más alto</strong> encarece la financiación y tiende a
      enfriar la demanda; uno <strong>más bajo</strong> la abarata y suele
      impulsar los precios. La dirección y la magnitud de la respuesta las
      determina el modelo a partir de lo aprendido en el periodo 2015-2024,
      que incluye tanto la etapa de tipos próximos a cero como el
      endurecimiento de 2022 y 2023.
    </div>
    <div class="deslizador">
      <span style="color:var(--suave);font-size:13px">−2 pp</span>
      <input type="range" id="s-tipo" min="-2" max="2" step="0.25" value="0">
      <span style="color:var(--suave);font-size:13px">+2 pp</span>
      <strong id="s-valor" style="min-width:92px;text-align:right;font-variant-numeric:tabular-nums">
        0,00 pp
      </strong>
    </div>
    <button class="boton" id="b-simular">Ejecutar el modelo</button>

    <div class="metricas" style="margin-top:24px">
      <div class="metrica">
        <div class="n" id="m-base">—</div>
        <div class="t">IAV medio base</div>
      </div>
      <div class="metrica">
        <div class="n" id="m-nuevo">—</div>
        <div class="t">IAV medio simulado</div>
      </div>
      <div class="metrica">
        <div class="n" id="m-dif">—</div>
        <div class="t">Variación</div>
      </div>
    </div>
  </div>

  <div class="tarjeta">
    <h2>Territorios más sensibles</h2>
    <p class="desc" id="e-resumen">Ejecute una simulación para ver el resultado.</p>
    <div style="max-height:440px;overflow-y:auto">
      <table id="t-escenario"></table>
    </div>
  </div>
</section>

</main>

<script>
const CARTO = {{ cartograma|tojson }};
const LADO = 52, HUECO = 5, MARGEN = 14;
let DATOS = [], MIN = 0, MAX = 1;
let FIJADA = null;

const fmt = (v, d = 2) =>
  v === null || v === undefined ? "—"
  : Number(v).toLocaleString("es-ES", {minimumFractionDigits: d, maximumFractionDigits: d});

/* Escala cromática divergente: azul (accesible) a rojo (tensionado) */
function color(v){
  const t = Math.max(0, Math.min(1, (v - MIN) / (MAX - MIN)));
  const paradas = [[44,127,184],[127,205,187],[254,217,118],[253,141,60],[227,26,28]];
  const p = t * (paradas.length - 1), i = Math.min(Math.floor(p), paradas.length - 2), f = p - i;
  const c = paradas[i].map((x, j) => Math.round(x + (paradas[i+1][j] - x) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function dibujarMapa(){
  const svg = document.getElementById("carto");
  svg.innerHTML = "";
  DATOS.forEach(d => {
    const celda = CARTO[d.provincia];
    if(!celda) return;
    const [fila, col] = celda;
    const x = MARGEN + col * (LADO + HUECO), y = MARGEN + fila * (LADO + HUECO);
    const fijada = FIJADA === d.provincia;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.innerHTML = `
      <rect class="celda" x="${x}" y="${y}" width="${LADO}" height="${LADO}"
            rx="7" fill="${color(d.IAV_previsto)}"
            stroke="${fijada ? '#ffffff' : '#0d1117'}"
            stroke-width="${fijada ? 3 : 1.5}"/>
      <text class="etiqueta" x="${x + LADO/2}" y="${y + LADO/2 - 1}">${d.abreviatura}</text>
      <text class="valor" x="${x + LADO/2}" y="${y + LADO/2 + 13}">${fmt(d.IAV_previsto, 1)}</text>`;

    /* El paso del cursor solo actualiza el panel si no hay
       ningún territorio fijado. */
    g.onmouseenter = () => { if(FIJADA === null) mostrarDetalle(d); };

    /* Al pinchar se fija el territorio; al volver a pinchar
       sobre el mismo, se libera. */
    g.onclick = () => {
      FIJADA = (FIJADA === d.provincia) ? null : d.provincia;
      dibujarMapa();
      if(GEO) dibujarMapaGeo();
      if(FIJADA) mostrarDetalle(d);
    };

    svg.appendChild(g);
  });
}

function mostrarDetalle(d){
  const fijada = FIJADA === d.provincia;
  document.getElementById("d-nombre").innerHTML =
    d.provincia + (fijada
      ? ' <span style="font-size:12px;color:var(--acento);font-weight:500">· fijado</span>'
      : '');
  document.getElementById("d-ccaa").textContent = fijada
    ? d.comunidad_autonoma + " — pulse de nuevo para liberar"
    : d.comunidad_autonoma;
  document.getElementById("d-iav").textContent = fmt(d.IAV_previsto);
  const signo = d.variacion_pct >= 0 ? "+" : "";
  document.getElementById("d-datos").innerHTML = `
    <div class="dato"><span>Intervalo del modelo</span>
      <span>${fmt(d.IAV_minimo)} – ${fmt(d.IAV_maximo)}</span></div>
    <div class="dato"><span>IAV en 2024</span><span>${fmt(d.IAV_2024)}</span></div>
    <div class="dato"><span>Variación prevista</span>
      <span style="color:${d.variacion_pct >= 0 ? 'var(--alerta)' : 'var(--acento2)'}">
        ${signo}${fmt(d.variacion_pct)}%</span></div>
    <div class="dato"><span>Precio previsto</span>
      <span>${fmt(d.precio_m2_previsto, 0)} €/m²</span></div>
    <div class="dato"><span>Vivienda de 80 m²</span>
      <span>${fmt(d.precio_m2_previsto * 80, 0)} €</span></div>
    <div class="dato"><span>Salario medio</span>
      <span>${fmt(d.salario_2024, 0)} €</span></div>
    <div class="dato"><span>Dispersión entre modelos</span>
      <span>±${fmt(d.dispersion, 3)}</span></div>`;
}
/* Navegación */
document.querySelectorAll("nav button").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach(x => x.classList.remove("activa"));
    document.querySelectorAll(".vista").forEach(x => x.classList.remove("activa"));
    b.classList.add("activa");
    document.getElementById(b.dataset.vista).classList.add("activa");
  };
});

/* Módulo 2 */
/* ---------- Mapa geográfico por cercanía ---------- */
const URL_GEO = "https://raw.githubusercontent.com/codeforgermany/" +
                "click_that_hood/main/public/data/spain-provinces.geojson";
let GEO = null, VISTA = "mapa";

/* Centro aproximado (lon, lat) de cada provincia. El emparejamiento entre
   cada polígono del archivo y su provincia se hace por proximidad a estos
   centros, de modo que resulta indiferente cómo esté escrito el nombre en
   el GeoJSON (Vizcaya, Bizkaia, etc.). */
const CENTROS = {
  "Coruña, A":[-8.4,43.0],"Lugo":[-7.4,43.0],"Ourense":[-7.6,42.2],
  "Pontevedra":[-8.5,42.4],"Asturias":[-6.0,43.3],"Cantabria":[-4.0,43.2],
  "Bizkaia":[-2.9,43.2],"Gipuzkoa":[-2.1,43.2],"Araba/Álava":[-2.8,42.8],
  "Navarra":[-1.6,42.7],"Rioja, La":[-2.5,42.3],"Huesca":[-0.1,42.2],
  "Zaragoza":[-1.1,41.5],"Teruel":[-0.9,40.6],"Lleida":[1.0,42.0],
  "Girona":[2.7,42.1],"Barcelona":[2.0,41.6],"Tarragona":[0.9,41.1],
  "Castellón/Castelló":[-0.1,40.2],"Valencia/València":[-0.7,39.4],
  "Alicante/Alacant":[-0.6,38.5],"Murcia":[-1.5,38.0],"Almería":[-2.3,37.2],
  "Granada":[-3.2,37.2],"Málaga":[-4.7,36.8],"Cádiz":[-5.8,36.5],
  "Sevilla":[-5.8,37.4],"Huelva":[-6.9,37.6],"Córdoba":[-4.8,38.0],
  "Jaén":[-3.4,38.0],"Badajoz":[-6.3,38.7],"Cáceres":[-6.1,39.7],
  "Toledo":[-4.2,39.8],"Ciudad Real":[-3.9,38.9],"Cuenca":[-2.2,39.9],
  "Guadalajara":[-2.6,40.8],"Madrid":[-3.7,40.4],"Ávila":[-5.0,40.6],
  "Segovia":[-4.0,41.0],"Soria":[-2.5,41.6],"Burgos":[-3.6,42.3],
  "Palencia":[-4.5,42.4],"Valladolid":[-4.8,41.6],"Zamora":[-6.0,41.7],
  "Salamanca":[-6.0,40.7],"León":[-5.8,42.6],
  "Balears, Illes":[2.9,39.6],"Palmas, Las":[-15.5,28.1],
  "Santa Cruz de Tenerife":[-16.6,28.3],"Ceuta":[-5.32,35.89],
  "Melilla":[-2.94,35.29],
};

function centroide(geom){
  let sx=0, sy=0, n=0;
  const anillos = geom.type==="Polygon" ? [geom.coordinates] : geom.coordinates;
  anillos.forEach(poly=>poly[0].forEach(c=>{ sx+=c[0]; sy+=c[1]; n++; }));
  return [sx/n, sy/n];
}

async function cargarGeo(){
  try{
    const r = await fetch(URL_GEO);
    if(!r.ok) throw 0;
    GEO = await r.json();
    dibujarMapaGeo();
  }catch(e){
    document.getElementById("aviso-mapa").textContent =
      "No se pudo cargar el mapa; se muestra el mosaico.";
    cambiarVista("mosaico");
    document.getElementById("b-mapa").style.display = "none";
  }
}

function dibujarMapaGeo(){
  const svg = document.getElementById("mapageo");
  const nombres = Object.keys(CENTROS);

  /* Proyección: península con las islas y ciudades autónomas recolocadas. */
  const W=1000, LON0=-9.6, LON1=4.4, LAT0=35.7, LAT1=44.0, H=760;
  function proyecta(lon, lat){
    if(lon < -12) return [(lon+18.5)*30+20, (29.5-lat)*30+H-40];      // Canarias
    if(lat < 36.0) return [(lon+8)*45+120, (36.4-lat)*70+H-70];       // Ceuta/Melilla
    return [(lon-LON0)/(LON1-LON0)*W, (LAT1-lat)/(LAT1-LAT0)*H];
  }

  let html="", usados={};
  GEO.features.forEach(f=>{
    const [clon,clat]=centroide(f.geometry);
    /* provincia más cercana al centro del polígono */
    let mejor=null, d=1e9;
    nombres.forEach(p=>{
      const dd=(CENTROS[p][0]-clon)**2 + (CENTROS[p][1]-clat)**2;
      if(dd<d){ d=dd; mejor=p; }
    });
    const dato = DATOS.find(x=>x.provincia===mejor);
    if(!dato) return;
    usados[mejor]=(usados[mejor]||0)+1;

    const anillos = f.geometry.type==="Polygon" ? [f.geometry.coordinates]
                                                : f.geometry.coordinates;
    let path="";
    anillos.forEach(poly=>poly.forEach(anillo=>{
      anillo.forEach((c,i)=>{
        const [x,y]=proyecta(c[0],c[1]);
        path += (i?"L":"M")+x.toFixed(1)+" "+y.toFixed(1);
      });
      path+="Z";
    }));
    const fij = FIJADA===mejor;
    html += `<path class="prov" d="${path}" fill="${color(dato.IAV_previsto)}"
             stroke="${fij?'#fff':'#0d1117'}" stroke-width="${fij?2.2:0.6}"
             data-prov="${mejor}"></path>`;
  });
  svg.innerHTML = html;

  svg.querySelectorAll("path").forEach(p=>{
    const dato = DATOS.find(x=>x.provincia===p.dataset.prov);
    p.onmouseenter = ()=>{ if(FIJADA===null) mostrarDetalle(dato); };
    p.onclick = ()=>{
      FIJADA = (FIJADA===dato.provincia) ? null : dato.provincia;
      dibujarMapaGeo(); dibujarMapa();
      if(FIJADA) mostrarDetalle(dato);
    };
  });

  const faltan = nombres.filter(p=>!usados[p]);
  document.getElementById("aviso-mapa").textContent =
    faltan.length ? "Sin representar: "+faltan.join(", ") : "";
}

function cambiarVista(v){
  VISTA=v;
  document.getElementById("mapageo").style.display = v==="mapa" ? "" : "none";
  document.getElementById("carto").style.display   = v==="mapa" ? "none" : "";
  document.getElementById("b-mapa").classList.toggle("activa", v==="mapa");
  document.getElementById("b-mosaico").classList.toggle("activa", v!=="mapa");
}
document.getElementById("b-mapa").onclick = ()=>cambiarVista("mapa");
document.getElementById("b-mosaico").onclick = ()=>cambiarVista("mosaico");

/* ---------- Alternancia entre modalidades de compra ---------- */
const modo = document.getElementById("modo-compra");
function alternarModo(){
  const contado = modo.value === "no";
  document.querySelectorAll(".solo-hipoteca").forEach(
    e => e.style.display = contado ? "none" : "");
  document.querySelectorAll(".solo-contado").forEach(
    e => e.style.display = contado ? "" : "none");
  document.getElementById("nota-esfuerzo").style.display = contado ? "none" : "";
}
modo.onchange = alternarModo;
alternarModo();

document.getElementById("b-calcular").onclick = async () => {
  const datos = {};
  new FormData(document.getElementById("f-perfil")).forEach((v, k) => datos[k] = v);
  new FormData(document.getElementById("f-supuestos")).forEach((v, k) => datos[k] = v);
  const contado = datos.hipoteca === "no";

  document.getElementById("p-resumen").textContent = "Calculando…";
  const r = await fetch("/api/perfil", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(datos)
  });
  const { resumen, territorios } = await r.json();
  territorios.sort((a, b) => a.precio_m2 - b.precio_m2);
  const viables = territorios.filter(f => f.viable).length;

  /* Resumen de capacidad de compra */
  let cabecera;
  if(contado){
    cabecera = `Con un presupuesto de <strong>${fmt(datos.presupuesto, 0)} €</strong> ` +
      `puede adquirir una vivienda de hasta <strong>${fmt(resumen.precio_maximo, 0)} €</strong> ` +
      `una vez descontados los gastos de compraventa.`;
  } else {
    cabecera = `La entidad admitiría una cuota de hasta ` +
      `<strong>${fmt(resumen.capacidad_mensual, 0)} € al mes</strong>, ` +
      `equivalente a un préstamo de <strong>${fmt(resumen.importe_financiable, 0)} €</strong>. ` +
      `Su capacidad de pago permite una vivienda de hasta ` +
      `${fmt(resumen.limite_por_capacidad, 0)} € y su ahorro alcanza para ` +
      `${fmt(resumen.limite_por_ahorro, 0)} €, de modo que el límite efectivo es de ` +
      `<strong>${fmt(resumen.precio_maximo, 0)} €</strong>.`;
  }

  document.getElementById("p-resumen").innerHTML = cabecera +
    `<br><br><strong>${viables}</strong> de 52 territorios permiten adquirir ` +
    `${datos.superficie} m². Los importes son medias provinciales e incluyen ` +
    `municipios pequeños, por lo que en las capitales serán superiores.`;

  const filas = territorios.map(f => contado ? `<tr>
      <td>${f.provincia}</td>
      <td>${fmt(f.precio_m2, 0)} €</td>
      <td>${fmt(f.coste, 0)} €</td>
      <td>${fmt(f.gastos, 0)} €</td>
      <td>${fmt(f.desembolso_inicial, 0)} €</td>
      <td>${f.falta_ahorro > 0
            ? '<span class="inviable">' + fmt(f.falta_ahorro, 0) + ' €</span>' : '—'}</td>
      <td>${fmt(f.superficie_asequible, 0)} m²</td>
      <td class="${f.viable ? 'viable' : 'inviable'}">${f.motivo}</td>
    </tr>` : `<tr>
      <td>${f.provincia}</td>
      <td>${fmt(f.precio_m2, 0)} €</td>
      <td>${fmt(f.coste, 0)} €</td>
      <td>${fmt(f.desembolso_inicial, 0)} €</td>
      <td>${f.falta_ahorro > 0
            ? '<span class="inviable">' + fmt(f.falta_ahorro, 0) + ' €</span>' : '—'}</td>
      <td>${fmt(f.cuota, 0)} €</td>
      <td>${f.esfuerzo === null ? "—" : fmt(f.esfuerzo, 1) + "%"}</td>
      <td>${fmt(f.superficie_asequible, 0)} m²</td>
      <td class="${f.viable ? 'viable' : 'inviable'}">${f.motivo}</td>
    </tr>`).join("");

  document.getElementById("t-perfil").innerHTML = (contado ? `
    <tr><th>Territorio</th><th>Precio m²</th><th>Precio vivienda</th>
        <th>Gastos</th><th>Desembolso total</th><th>Le falta</th>
        <th>m² asequibles</th><th>Situación</th></tr>` : `
    <tr><th>Territorio</th><th>Precio m²</th><th>Precio vivienda</th>
        <th>Desembolso inicial</th><th>Le falta</th><th>Cuota mensual</th>
        <th>Esfuerzo</th><th>m² asequibles</th><th>Situación</th></tr>`) + filas;
};

/* Módulo 3 */
const deslizador = document.getElementById("s-tipo");
deslizador.oninput = () => {
  const v = Number(deslizador.value);
  document.getElementById("s-valor").textContent =
    (v >= 0 ? "+" : "") + fmt(v) + " pp";
};

document.getElementById("b-simular").onclick = async () => {
  document.getElementById("e-resumen").textContent = "Ejecutando los siete modelos…";
  const r = await fetch("/api/escenario", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({variacion: Number(deslizador.value)})
  });
  const d = await r.json();

  document.getElementById("m-base").textContent = fmt(d.media_base);
  document.getElementById("m-nuevo").textContent = fmt(d.media);
  const s = d.variacion_media >= 0 ? "+" : "";
  document.getElementById("m-dif").textContent = s + fmt(d.variacion_media);

  const filas = d.detalle.sort((a, b) =>
    Math.abs(b.diferencia) - Math.abs(a.diferencia)).slice(0, 20);

  document.getElementById("e-resumen").innerHTML =
    `Veinte territorios con mayor respuesta a una variación de ` +
    `<strong>${s}${fmt(deslizador.value)} pp</strong> en el tipo de interés.`;

  document.getElementById("t-escenario").innerHTML = `
    <tr><th>Territorio</th><th>IAV base</th><th>IAV simulado</th><th>Diferencia</th></tr>` +
    filas.map(f => `<tr>
      <td>${f.provincia}</td>
      <td>${fmt(f.IAV_previsto)}</td>
      <td>${fmt(f.IAV)}</td>
      <td style="color:${f.diferencia >= 0 ? 'var(--alerta)' : 'var(--acento2)'}">
        ${f.diferencia >= 0 ? "+" : ""}${fmt(f.diferencia, 3)}</td>
    </tr>`).join("");
};

/* Arranque */
fetch("/api/mapa").then(r => r.json()).then(d => {
  DATOS = d;
  const valores = d.map(x => x.IAV_previsto);
  MIN = Math.min(...valores); MAX = Math.max(...valores);
  dibujarMapa();
  cargarGeo();
  mostrarDetalle(d.find(x => x.provincia === "Madrid") || d[0]);
});
</script>
</body>
</html>
"""


# ==============================================================================
# 7. ARRANQUE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("APLICACIÓN DE ACCESIBILIDAD RESIDENCIAL — TFM")
    print("=" * 70)
    print(f"Territorios cargados : {len(PREDICCIONES)}")
    print(f"Modelos combinados   : {len(PAQUETE['modelos'])}")
    print(f"Año predicho         : {PAQUETE['anio_prediccion']}")
    print()
    print("Abra en el navegador:  http://127.0.0.1:5000")
    print("Para detener la aplicación, pulse Ctrl+C en la consola.")
    print("=" * 70)

    app.run(debug=False, port=5000)
