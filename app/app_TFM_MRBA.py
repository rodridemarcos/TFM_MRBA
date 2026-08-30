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

# ==============================================================================
# 1. PARÁMETROS Y SUPUESTOS
# ==============================================================================

DIRECTORIO = Path(__file__).parent

# Supuestos hipotecarios. NO proceden del modelo: son valores por defecto
# de la aplicación, editables por el usuario desde la interfaz.
SUPUESTOS = {
    "plazo_anios": 30,
    "entrada_pct": 20.0,
    "tipo_interes_pct": 3.2,      # Último dato del Banco de España
    "gastos_compra_pct": 10.0,    # Impuestos y gastos de formalización
    "esfuerzo_maximo_pct": 35.0,  # Umbral prudencial de esfuerzo
}

# ==============================================================================
# 2. CARTOGRAMA DE MOSAICO
# ==============================================================================

# Cada territorio ocupa una celda en una retícula que reproduce de forma
# esquemática la geografía peninsular. A diferencia de un mapa real, todas
# las provincias reciben la misma superficie, de modo que las de menor
# extensión resultan igual de legibles.

CARTOGRAMA = {
    "Coruña, A": (0, 2), "Lugo": (0, 3), "Asturias": (0, 4),
    "Cantabria": (0, 5), "Bizkaia": (0, 6), "Gipuzkoa": (0, 7),

    "Pontevedra": (1, 2), "Ourense": (1, 3), "León": (1, 4),
    "Palencia": (1, 5), "Burgos": (1, 6), "Araba/Álava": (1, 7),
    "Navarra": (1, 8), "Huesca": (1, 9), "Girona": (1, 10),

    "Zamora": (2, 3), "Valladolid": (2, 4), "Soria": (2, 6),
    "Rioja, La": (2, 7), "Zaragoza": (2, 8), "Lleida": (2, 9),
    "Barcelona": (2, 10),

    "Salamanca": (3, 3), "Ávila": (3, 4), "Segovia": (3, 5),
    "Guadalajara": (3, 6), "Teruel": (3, 8), "Tarragona": (3, 9),

    "Cáceres": (4, 3), "Madrid": (4, 5), "Cuenca": (4, 6),
    "Castellón/Castelló": (4, 8),

    "Badajoz": (5, 3), "Toledo": (5, 5), "Valencia/València": (5, 7),

    "Ciudad Real": (6, 4), "Albacete": (6, 6),
    "Alicante/Alacant": (6, 7),

    "Huelva": (7, 2), "Sevilla": (7, 3), "Córdoba": (7, 4),
    "Jaén": (7, 5), "Murcia": (7, 7),

    "Cádiz": (8, 3), "Málaga": (8, 4), "Granada": (8, 5),
    "Almería": (8, 6),

    "Palmas, Las": (9, 0), "Santa Cruz de Tenerife": (9, 1),
    "Ceuta": (9, 3), "Melilla": (9, 4), "Balears, Illes": (9, 9),
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


def calcular_perfil(datos):
    """Aplica la situación personal del usuario sobre la salida del modelo.

    Ninguno de estos valores entra en el modelo. El IAV personal resulta de
    sustituir el salario provincial por el del usuario y la superficie de
    referencia por la solicitada.
    """
    ingresos = float(datos["ingresos"])
    superficie = float(datos["superficie"])
    ahorros = float(datos["ahorros"])
    aportacion_anual = float(datos["aportacion_anual"])
    sexo = datos.get("sexo", "total")
    con_hipoteca = datos.get("hipoteca", "si") == "si"

    plazo = float(datos.get("plazo", SUPUESTOS["plazo_anios"]))
    entrada_pct = float(datos.get("entrada", SUPUESTOS["entrada_pct"]))
    tipo = float(datos.get("tipo", SUPUESTOS["tipo_interes_pct"]))
    gastos_pct = SUPUESTOS["gastos_compra_pct"]

    filas = []

    for _, provincia in PREDICCIONES.iterrows():
        precio_m2 = provincia["precio_m2_previsto"]
        coste = precio_m2 * superficie
        coste_total = coste * (1 + gastos_pct / 100)

        # IAV de referencia según el sexo, cuando procede.
        if sexo == "hombre":
            salario_referencia = provincia["salario_hombre_2024"]
        elif sexo == "mujer":
            salario_referencia = provincia["salario_mujer_2024"]
        else:
            salario_referencia = provincia["salario_2024"]

        iav_referencia = (
            precio_m2 * superficie / salario_referencia
        )
        iav_personal = coste / ingresos

        entrada_necesaria = (
            coste * entrada_pct / 100 + coste * gastos_pct / 100
        )
        falta_ahorro = max(entrada_necesaria - ahorros, 0)
        anios_ahorro = (
            falta_ahorro / aportacion_anual
            if aportacion_anual > 0 else np.inf
        )

        if con_hipoteca:
            principal = coste - coste * entrada_pct / 100
            interes_mensual = tipo / 100 / 12
            meses = plazo * 12

            if interes_mensual > 0:
                factor = (1 + interes_mensual) ** meses
                cuota = principal * interes_mensual * factor / (factor - 1)
            else:
                cuota = principal / meses
        else:
            principal = 0.0
            cuota = 0.0

        esfuerzo = cuota * 12 / ingresos * 100 if ingresos > 0 else np.inf

        viable = (
            con_hipoteca
            and esfuerzo <= SUPUESTOS["esfuerzo_maximo_pct"]
            and np.isfinite(anios_ahorro)
            and anios_ahorro <= 10
        ) or (not con_hipoteca and coste_total <= ahorros)

        filas.append({
            "provincia": provincia["provincia"],
            "iav_provincial": round(provincia["IAV_previsto"], 2),
            "iav_referencia": round(iav_referencia, 2),
            "iav_personal": round(iav_personal, 2),
            "precio_m2": round(precio_m2, 0),
            "coste": round(coste, 0),
            "coste_total": round(coste_total, 0),
            "entrada_necesaria": round(entrada_necesaria, 0),
            "anios_ahorro": (
                round(anios_ahorro, 1)
                if np.isfinite(anios_ahorro) else None
            ),
            "cuota": round(cuota, 0),
            "esfuerzo": round(esfuerzo, 1) if np.isfinite(esfuerzo) else None,
            "viable": bool(viable),
        })

    return filas


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
    <button data-vista="perfil">Perfil personal</button>
    <button data-vista="escenario">Simulador de escenarios</button>
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
      <svg id="carto" viewBox="0 0 655 598"></svg>
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
    <h2>Situación personal</h2>
    <p class="desc">
      Estos datos no intervienen en el modelo. Se aplican sobre el precio
      previsto por este para calcular su situación particular.
    </p>
    <form id="f-perfil">
      <div>
        <label>Salario bruto anual (€)</label>
        <input type="number" name="ingresos" value="28000" min="1000" step="500">
      </div>
      <div>
        <label>Superficie buscada (m²)</label>
        <input type="number" name="superficie" value="80" min="20" max="400">
      </div>
      <div>
        <label>Ahorros disponibles (€)</label>
        <input type="number" name="ahorros" value="30000" min="0" step="1000">
      </div>
      <div>
        <label>Ahorro anual previsto (€)</label>
        <input type="number" name="aportacion_anual" value="6000" min="0" step="500">
      </div>
      <div>
        <label>Referencia salarial territorial</label>
        <select name="sexo">
          <option value="total">Media provincial</option>
          <option value="hombre">Salario masculino</option>
          <option value="mujer">Salario femenino</option>
        </select>
      </div>
      <div>
        <label>Financiación</label>
        <select name="hipoteca">
          <option value="si">Con hipoteca</option>
          <option value="no">Sin hipoteca</option>
        </select>
      </div>
    </form>

    <h2 style="margin-top:26px">Supuestos hipotecarios</h2>
    <div class="nota">
      Los siguientes valores son <strong>supuestos de la aplicación</strong> y
      no resultados del modelo. Puede modificarlos libremente.
    </div>
    <form id="f-supuestos">
      <div>
        <label>Plazo (años)</label>
        <input type="number" name="plazo" value="{{ supuestos.plazo_anios }}" min="5" max="40">
      </div>
      <div>
        <label>Entrada (%)</label>
        <input type="number" name="entrada" value="{{ supuestos.entrada_pct }}" min="0" max="100" step="5">
      </div>
      <div>
        <label>Tipo de interés (%)</label>
        <input type="number" name="tipo" value="{{ supuestos.tipo_interes_pct }}" min="0" max="15" step="0.1">
      </div>
    </form>

    <div style="margin-top:22px">
      <button class="boton" id="b-calcular">Calcular</button>
    </div>
  </div>

  <div class="tarjeta">
    <h2>Resultado por territorio</h2>
    <p class="desc" id="p-resumen">Introduzca sus datos y pulse Calcular.</p>
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
      El análisis de interpretabilidad identificó el tipo de interés hipotecario
      como la variable determinante del modelo: su retirada duplicaba el error.
      Modifique su valor y el modelo recalculará la previsión de los 52
      territorios.
    </p>
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
document.getElementById("b-calcular").onclick = async () => {
  const datos = {};
  new FormData(document.getElementById("f-perfil")).forEach((v, k) => datos[k] = v);
  new FormData(document.getElementById("f-supuestos")).forEach((v, k) => datos[k] = v);

  document.getElementById("p-resumen").textContent = "Calculando…";
  const r = await fetch("/api/perfil", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(datos)
  });
  const filas = await r.json();
  filas.sort((a, b) => a.iav_personal - b.iav_personal);

  const viables = filas.filter(f => f.viable).length;
  document.getElementById("p-resumen").innerHTML =
    `<strong>${viables}</strong> de 52 territorios resultan viables con su situación. ` +
    `Ordenados de menor a mayor esfuerzo.`;

  document.getElementById("t-perfil").innerHTML = `
    <tr><th>Territorio</th><th>Años de su salario</th><th>Coste total</th>
        <th>Entrada</th><th>Años de ahorro</th><th>Cuota mensual</th>
        <th>Esfuerzo</th><th>Viable</th></tr>` +
    filas.map(f => `<tr>
      <td>${f.provincia}</td>
      <td>${fmt(f.iav_personal)}</td>
      <td>${fmt(f.coste_total, 0)} €</td>
      <td>${fmt(f.entrada_necesaria, 0)} €</td>
      <td>${f.anios_ahorro === null ? "—" : fmt(f.anios_ahorro, 1)}</td>
      <td>${fmt(f.cuota, 0)} €</td>
      <td>${f.esfuerzo === null ? "—" : fmt(f.esfuerzo, 1) + "%"}</td>
      <td class="${f.viable ? 'viable' : 'inviable'}">${f.viable ? "Sí" : "No"}</td>
    </tr>`).join("");
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
