# Proceso de Extracción de Datos RAW - OHLCV

Fecha de ejecución: 2025-10-25
Última actualización: 2025-10-25
Versión: 3.0 (Solo Análisis Técnico)
Estado: COMPLETADO

---

## Objetivo

Extraer datos históricos de precios OHLCV (Open, High, Low, Close, Volume) de 10 acciones representativas del S&P 500 para realizar análisis técnico y construir modelos de clasificación binaria. Esta extracción NO aplica transformaciones, guardando los datos tal cual vienen de la API de Yahoo Finance.

## Fuente de Datos

Utilizamos yfinance (Yahoo Finance Python Library) para obtener los datos. Esta decisión se basó en que es una fuente pública, no requiere autenticación, y proporciona series temporales de precios diarios con actualización diaria durante días de mercado abierto.

Documentación oficial: https://pypi.org/project/yfinance/

### Por qué solo Yahoo Finance

Inicialmente consideramos combinar datos técnicos (yfinance) con datos fundamentales (Financial Modeling Prep API), pero identificamos una limitación crítica: las fuentes gratuitas solo proporcionan 5-6 trimestres recientes (~1.5 años de datos). Con solo 5-6 observaciones por ticker, no es posible hacer un train/test split adecuado para entrenar modelos robustos.

Por esta razón, decidimos enfocar el proyecto exclusivamente en análisis técnico con 10 años completos de datos OHLCV (2015-2024), lo cual proporciona ~2,500 observaciones por ticker. Esto indica que sacrificamos la combinación de análisis técnico-fundamental a cambio de un volumen de datos suficiente para entrenamiento supervisado de alta calidad.

## Datos Extraídos - Resumen

### Extracción Exitosa

| Métrica | Valor |
|---------|-------|
| Tickers procesados | 10/10 (100% éxito) |
| Total filas OHLCV | 85,642 días |
| Archivos generados | 10 archivos Parquet |
| Periodo histórico | Desde 1962 hasta 2025-10-24 (máximo disponible por ticker) |
| Periodo utilizado | 2015-01-01 a 2024-12-31 (filtrado en fase Transform) |

Se observa que el volumen de datos extraído (85,642 días) es 2.4 veces superior a lo inicialmente planeado (~35,000 días), lo cual es excelente para la robustez del modelo. Esto se debe a que yfinance proporciona históricos muy extensos (hasta 63 años para XOM).

## Tickers Seleccionados y Justificación

### Criterios de Selección

Los 10 tickers fueron seleccionados bajo los siguientes criterios:

- Diversificación sectorial: Cobertura de 6 sectores diferentes del S&P 500
- Capitalización de mercado: Alta capitalización (large-cap stocks)
- Liquidez: Alto volumen de transacciones diarias
- Histórico disponible: Mínimo 10 años de datos completos
- Mezcla de perfiles: Combinación de growth stocks (NVDA, TSLA) y value stocks (JPM, XOM)

### Tabla Detallada por Ticker

| Ticker | Empresa | Sector | Días Históricos | Desde | Hasta | Años de Datos |
|--------|---------|--------|----------------|-------|-------|---------------|
| XOM | Exxon Mobil | Energy | 16,062 | 1962-01-02 | 2025-10-24 | 63 años |
| JPM | JPMorgan | Financial | 11,497 | 1980-03-17 | 2025-10-24 | 45 años |
| AAPL | Apple | Technology | 11,309 | 1980-12-12 | 2025-10-24 | 44 años |
| UNH | UnitedHealth | Healthcare | 10,336 | 1984-10-17 | 2025-10-24 | 40 años |
| MSFT | Microsoft | Technology | 9,983 | 1986-03-13 | 2025-10-24 | 39 años |
| AMZN | Amazon | Consumer Cyclical | 7,157 | 1997-05-15 | 2025-10-24 | 28 años |
| NVDA | NVIDIA | Technology | 6,732 | 1999-01-22 | 2025-10-24 | 26 años |
| GOOGL | Alphabet | Communication | 5,331 | 2004-08-19 | 2025-10-24 | 21 años |
| TSLA | Tesla | Consumer Cyclical | 3,856 | 2010-06-29 | 2025-10-24 | 15 años |
| META | Meta | Communication | 3,379 | 2012-05-18 | 2025-10-24 | 13 años |

Total: 85,642 días de datos OHLCV

### Interpretación de la Cobertura Sectorial

| Sector | Tickers | % del Total | Observación |
|--------|---------|-------------|-------------|
| Technology | AAPL, MSFT, NVDA | 30% | Sector más representado, refleja importancia actual |
| Communication | GOOGL, META | 20% | Empresas de alto crecimiento en última década |
| Consumer Cyclical | AMZN, TSLA | 20% | Mix de e-commerce y manufactura |
| Financial | JPM | 10% | Sector bancario tradicional |
| Healthcare | UNH | 10% | Sector defensivo |
| Energy | XOM | 10% | Sector cíclico, histórico más largo |

Esta distribución permite que el modelo aprenda patrones de comportamiento técnico en diferentes sectores económicos, lo que mejora su capacidad de generalización.

## Datos Técnicos (OHLCV)

### Configuración de Extracción

```python
# Configuración utilizada en src/etl/extract.py
PERIOD = 'max'      # Histórico completo disponible
INTERVAL = '1d'     # Velas diarias (no intraday)
FORMAT = 'parquet'  # Formato Parquet con compresión Snappy
```

### Estructura de Datos

Cada archivo Parquet contiene las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| Ticker | string | Símbolo de la acción (ej: AAPL) |
| Date | datetime (UTC) | Fecha de la vela diaria |
| Open | float | Precio de apertura del día |
| High | float | Precio máximo alcanzado en el día |
| Low | float | Precio mínimo alcanzado en el día |
| Close | float | Precio de cierre del día |
| Volume | int64 | Volumen de transacciones |
| Dividends | float | Dividendos pagados (si aplica) |
| Stock Splits | float | Splits de acciones (si aplica) |

Nota importante: Las columnas Dividends y Stock Splits se preservan en los datos RAW, pero NO se utilizan en el análisis técnico. Solo se usan para registro histórico.

### Archivos Generados

```
data/raw_parquet/
├── AAPL_ohlcv.parquet
├── MSFT_ohlcv.parquet
├── NVDA_ohlcv.parquet
├── GOOGL_ohlcv.parquet
├── AMZN_ohlcv.parquet
├── META_ohlcv.parquet
├── TSLA_ohlcv.parquet
├── JPM_ohlcv.parquet
├── UNH_ohlcv.parquet
└── XOM_ohlcv.parquet
```


Se observa que el formato Parquet con compresión Snappy es extremadamente eficiente, reduciendo el tamaño de almacenamiento en comparación con CSV tradicional.

## Características de la Extracción RAW

### Sin Transformaciones

Esta extracción no aplica ninguna transformación: timestamps en UTC originales, NaN/None preservados, todos los días de mercado incluidos, 
No hay validaciones de datos ni limpiezas.

### Razón del Enfoque RAW

Con este enfoque mantenemos procesos diferenciados asegurandonos de que todo sigue un pipeline estrcuturado: extracción, transformación, análisis. Así aseguramos procesos independientes que en caso de uno de ellos ser modificado, por ejemplo la transformación, la carga se mantiene igual.

## Metadata Generada

### Archivo de Metadata

Ubicación: `data/metadata/extraction_log.yaml`

Contenido generado:

```yaml
extraction_date: '2025-10-25T17:50:44.748077Z'
config:
  period: max
  interval: 1d
  source: yfinance
  data_type: OHLCV
results:
  total_tickers: 10
  total_days: 85642
  total_files: 10
  tickers:
    AAPL:
      rows: 11309
      date_from: '1980-12-12'
      date_to: '2025-10-24'
      file: AAPL_ohlcv.parquet
    # ... (similar para otros tickers)
status: success
```

Este archivo de metadata permite rastrear exactamente qué datos se extrajeron, cuándo y con qué configuración. Esto es esencial para versionamiento con DVC y reproducibilidad.

## Limitaciones y Consideraciones

### 1. Datos Fundamentales Descartados

Limitación identificada:
- Solo 5-6 trimestres disponibles en fuentes gratuitas (~1.5 años)
- Insuficiente para train/test split robusto en ML

Decisión tomada:
- Enfocar proyecto en análisis técnico exclusivamente
- Utilizar 10 años completos de OHLCV (2015-2024)

### 2. Volatilidad de la API Yahoo Finance

Limitación conocida:
- Yahoo Finance puede estar temporalmente no disponible
- Rate limits informales (no documentados oficialmente)
- Estructura de datos puede cambiar sin aviso

Mitigación aplicada:
- Datos ya extraídos y guardados localmente en Parquet
- Versionamiento con DVC permitirá recuperar datos históricos
- Metadata permite rastrear fuente y fecha de extracción

---

Última revisión: 2025-10-25
Autor: Grupo 27 - Universidad de Los Andes
Proyecto: Sistema de Clasificación de Acciones S&P 500 (Análisis Técnico)
