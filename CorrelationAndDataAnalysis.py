import pandas as pd
import numpy as np

def calcular_correlaciones_y_errores(archivo1, archivo2, num_filas=None):
    # Cargar los archivos CSV de manera eficiente
    df1 = pd.read_csv(archivo1, nrows=num_filas + 1 if num_filas else None)
    df2 = pd.read_csv(archivo2, nrows=num_filas + 1 if num_filas else None)

    # Limitar el número de filas si se especifica (excluyendo la fila de títulos)
    df1 = df1.iloc[1:]
    df2 = df2.iloc[1:]

    # Verificar que los archivos tengan las mismas columnas
    columnas_comunes = df1.columns.intersection(df2.columns)
    if columnas_comunes.empty:
        raise ValueError("No hay columnas en común entre los dos archivos.")

    resultados = {}

    for columna in columnas_comunes:
        datos1 = df1[columna].dropna().values
        datos2 = df2[columna].dropna().values

        min_length = min(len(datos1), len(datos2))
        datos1, datos2 = datos1[:min_length], datos2[:min_length]

        if min_length > 1:
            correlacion = round(np.corrcoef(datos1, datos2)[0, 1], 3)
        else:
            correlacion = np.nan

        error = np.nan
        if min_length > 2 and correlacion != 0:
            error = round((np.sqrt((1 - correlacion ** 2) / (min_length - 2)) / abs(correlacion)) * 100, 3)

        media1, media2 = round(np.mean(datos1), 3), round(np.mean(datos2), 3)
        std1, std2 = round(np.std(datos1, ddof=1), 3), round(np.std(datos2, ddof=1), 3)
        covarianza = round(np.cov(datos1, datos2)[0, 1], 3)
        r_cuadrado = round(correlacion ** 2, 3) if not np.isnan(correlacion) else np.nan

        resultados[columna] = {
            "correlacion": correlacion,
            "error_relativo": error,
            "media_1": media1,
            "media_2": media2,
            "desviacion_1": std1,
            "desviacion_2": std2,
            "covarianza": covarianza,
            "r_cuadrado": r_cuadrado
        }

    def calcular_radios_y_fases(x, y):
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            return np.nan, [], []
        angulos = np.unwrap(np.arctan2(y, x))
        cambios = np.diff(angulos)
        vueltas = round(np.sum(cambios) / (2 * np.pi), 3)
        radios = np.round(np.sqrt(x ** 2 + y ** 2), 3)
        return abs(vueltas), radios, angulos

    vueltas_archivo1, radios_archivo1, fases_archivo1 = calcular_radios_y_fases(df1.iloc[:, 1].dropna().values, df1.iloc[:, 2].dropna().values)
    vueltas_archivo2, radios_archivo2, fases_archivo2 = calcular_radios_y_fases(df2.iloc[:, 1].dropna().values, df2.iloc[:, 2].dropna().values)

    resultados["vueltas_y_radios"] = {
        "vueltas_archivo1": vueltas_archivo1,
        "vueltas_archivo2": vueltas_archivo2,
        "radios_archivo1": radios_archivo1,
        "radios_archivo2": radios_archivo2
    }

    if len(radios_archivo1) > 1 and len(radios_archivo2) > 1:
        min_length = min(len(radios_archivo1), len(radios_archivo2))
        correlacion_radios = round(np.corrcoef(radios_archivo1[:min_length], radios_archivo2[:min_length])[0, 1], 3)
    else:
        correlacion_radios = np.nan
    resultados["correlacion_radios"] = correlacion_radios

    if len(fases_archivo1) > 1 and len(fases_archivo2) > 1:
        min_length = min(len(fases_archivo1), len(fases_archivo2))
        desfase = np.mean(np.abs(fases_archivo1[:min_length] - fases_archivo2[:min_length]))
        desfase = round(desfase, 3)
    else:
        desfase = np.nan
    resultados["desfase_fases"] = desfase

    return resultados


archivo1 = "graph_results.csv"
archivo2 = "simulation_results.csv"
num_filas = 1048575

resultados = calcular_correlaciones_y_errores(archivo1, archivo2, num_filas)
for columna, valores in resultados.items():
    print(f"Columna: {columna}")
    if columna == "vueltas_y_radios":
        print(f"  Vueltas en archivo 1: {valores['vueltas_archivo1']}")
        print(f"  Vueltas en archivo 2: {valores['vueltas_archivo2']}")
    elif columna == "correlacion_radios":
        print(f"  Correlación entre radios: {valores}")
    elif columna == "desfase_fases":
        print(f"  Desfase promedio entre trayectorias: {valores} radianes")
    else:
        print(f"  Correlación: {valores['correlacion']}")
        print(f"  Error relativo porcentual: {valores['error_relativo']}%")
        print(f"  Media Archivo 1: {valores['media_1']}")
        print(f"  Media Archivo 2: {valores['media_2']}")
        print(f"  Desviación estándar Archivo 1: {valores['desviacion_1']}")
        print(f"  Desviación estándar Archivo 2: {valores['desviacion_2']}")
        print(f"  Covarianza: {valores['covarianza']}")
        print(f"  Coeficiente de determinación (R²): {valores['r_cuadrado']}")