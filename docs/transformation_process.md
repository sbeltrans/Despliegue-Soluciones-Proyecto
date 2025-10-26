# Proceso de Transformación de Datos - Feature Engineering

Fecha de ejecución: 2025-10-25
Última actualización: 2025-10-26
Versión: 1.0
Estado: COMPLETADO

---

## Objetivo

Transformar los datos RAW de OHLCV en features ML-ready para clasificación binaria (Comprar vs Vender). Este proceso incluye el cálculo de indicadores técnicos, creación de features adicionales, definición de la variable Target, y preparación final de datos para modelos de machine learning.

## Pipeline de Transformación

### Orden de Ejecución

El pipeline de transformación sigue este orden específico:

1. Cargar datos RAW (histórico completo)
2. Calcular indicadores técnicos sobre histórico completo
3. Calcular features adicionales
4. Crear variable Target (Y)
5. Filtrar fechas a 2015-2024
6. Eliminar columnas de precio absoluto
7. Eliminar NaN
8. Guardar features ML-ready

## Resultados del Proceso de Transformación

### Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Input: Datos RAW | 85,642 días (histórico completo) |
| Output: Features procesadas | 25,160 días (2015-2024) |
| Reducción de datos | 70.6% (filtrado temporal) |
| Features generadas | 17 indicadores técnicos |
| Variable Target | Y (binaria: 1=Comprar, 0=Vender) |
| Balance de clases | 52.8% Comprar / 47.2% Vender |
| Archivos generados | 10 archivos individuales + 1 consolidado |

Se observa que el balance de clases es excelente (52.8% / 47.2%), lo que indica que NO será necesario aplicar técnicas de balanceo (oversampling/undersampling) en la fase de modelado. 

## Decisión 1: Por qué calcular indicadores sobre histórico completo

### Problema Técnico

Los indicadores técnicos necesitan ventanas de datos pasados para calcularse correctamente:

- SMA_50: Necesita 50 días previos
- EMA_26: Necesita ~26 días previos (con ponderación exponencial)
- RSI_14: Necesita 14 días previos
- MACD: Necesita hasta 26 días previos
- Bollinger Bands: Necesita 20 días previos

### Qué pasaría si filtramos ANTES de calcular indicadores

Escenario problemático: Los límites inferiores de los datos no tienen información suficiente para crear los indicadores.

Resultado:
- Fecha 2015-01-01: SMA_50 = NaN (no hay 50 días previos)
- Fecha 2015-01-02: SMA_50 = NaN
- ...
- Fecha 2015-02-19: SMA_50 = NaN
- Fecha 2015-02-20: SMA_50 = valor (primer valor válido después de 50 días)

Esto indica que perderíamos casi 2 meses de datos al inicio de 2015 por NaN en los indicadores.

### Solución Aplicada

Creamos los indicadores y luego filtramos, así garantizamos que están todos los datos necesarios.

Resultado:
- Fecha 2015-01-01: SMA_50 = valor válido (calculado con datos de 2014)
- Fecha 2015-01-02: SMA_50 = valor válido
- Sin pérdida de datos en 2015

### Justificación de Negocio

Para trading algorítmico, es fundamental tener indicadores válidos desde el primer día del periodo de análisis. Si el modelo solo puede empezar a operar desde marzo 2015 (por NaN), perderíamos oportunidades de trading en enero-febrero 2015.

Adicionalmente, para el train/test split temporal (80/20), necesitamos maximizar los datos disponibles. Perder 2 meses al inicio reduce el conjunto de entrenamiento de manera innecesaria.

## Decisión 2: Por qué eliminar precio absoluto de las features

### Problema de Generalización

Contexto: Tenemos 10 tickers con rangos de precios muy diferentes:

| Ticker | Precio Promedio (2015-2024) | Rango Aproximado |
|--------|----------------------------|------------------|
| TSLA | $50 - $400 | Alta volatilidad |
| AAPL | $30 - $200 | Moderada (ajustado por splits) |
| NVDA | $10 - $500 | Muy alta volatilidad |
| AMZN | $300 - $3,500 | Rango muy amplio |
| GOOGL | $50 - $150 | Moderado |
| META | $50 - $380 | Alta volatilidad |

### Qué pasaría si incluimos precio absoluto

- El modelo aprende que "Close > 300 → Comprar" (basado en AMZN)
- Pero aplica esa regla a AAPL (Close ~$150) → Predicción incorrecta
- El modelo NO puede generalizar entre tickers porque aprende rangos de precios específicos

### Solución Aplicada

Features relativas (NO absolutas):

- Eliminadas: Open, High, Low, Close, Volume
- Mantenidas: Indicadores que ya son relativos o normalizados:
  - RSI_14: Normalizado 0-100
  - Returns: Cambio porcentual
  - SMA_20, SMA_50: Promedio móvil
  - MACD, MACD_signal: Diferencias de EMAs
  - BB_width: Ancho relativo de Bandas de Bollinger
  - Volatility_10: Desviación estándar de returns
  - Volume_change: Cambio porcentual de volumen

### Justificación de Negocio

En análisis técnico, lo que importa no es el precio absoluto, sino:

- Tendencia: ¿El precio está subiendo o bajando? → Returns
- Momentum: ¿Qué tan fuerte es el movimiento? → RSI, MACD
- Volatilidad: ¿Qué tan inestable está el precio? → ATR, BB_width
- Volumen: ¿Aumentó o disminuyó el interés? → Volume_change, OBV

Ninguno de estos conceptos depende del precio absoluto. Un movimiento de +2% es alcista tanto para AAPL como para AMZN independientemente del precio en el que encuentren.

## Decisión 3: Por qué NO normalizar ni estandarizar las features

### Decisión Tomada

No normalizar en el pipeline de transformación por las siguientes razones:

1. Flexibilidad: Dejar la normalización como un paso opcional para el modelado.
2. Modelos basados ene árboles: Random Forest y XGBoost no necesitan normalización y suelen tener mejor performance en datos reales.
3. Interpretabilidad: Features sin normalizar son más fáciles de interpretar (ej: RSI=70 significa sobrecompra).
4. Evitar data leakage: La normalización debe calcularse solo con datos de train, no con todo el dataset.

## Decisión 4: Por qué filtrar a 2015-2024

### Análisis Temporal de Disponibilidad de Datos

Datos disponibles por ticker:

| Ticker | Desde | Hasta | Años Disponibles | Años Utilizados (2015-2024) |
|--------|-------|-------|------------------|----------------------------|
| XOM | 1962 | 2025 | 63 | 10 |
| JPM | 1980 | 2025 | 45 | 10 |
| AAPL | 1980 | 2025 | 44 | 10 |
| UNH | 1984 | 2025 | 40 | 10 |
| MSFT | 1986 | 2025 | 39 | 10 |
| AMZN | 1997 | 2025 | 28 | 10 |
| NVDA | 1999 | 2025 | 26 | 10 |
| GOOGL | 2004 | 2025 | 21 | 10 |
| TSLA | 2010 | 2025 | 15 | 10 |
| META | 2012 | 2025 | 13 | 10 |

### Problema de Datos Desbalanceados por Ticker

Si utilizamos todos los datos tendríamos un desbalance de tickers, lo que perjudicaría el desarrollo de modelo, pues estaría sesgado a aquellos que tienen datos.

### Solución: Ventana Temporal Común (2015-2024)

Razones para elegir 2015-2024:

1. Balance entre tickers: Todos los 10 tickers tienen datos completos en este periodo
2. Relevancia: Mercados pre-2015 tienen dinámicas muy diferentes (pre-crisis 2008, pre-QE, etc.)
3. Volumen suficiente: ~2,500 días por ticker (10 años) es suficiente para train/test split
4. Datos recientes: Incluye eventos relevantes (COVID-2020, bull market 2020-2021, corrección 2022)

### Justificación de Negocio

Para un modelo de clasificación que predice movimientos de corto plazo (siguiente día), es más importante tener datos recientes y balanceados que históricos extensos de algunos tickers. Las dinámicas del mercado en 1980 (trading manual, sin algoritmos HFT, sin ETFs) son muy diferentes a las de 2015-2024.

## Decisión 5: Variable Target - Por qué este approach

### Pregunta de Negocio

Objetivo del modelo: Predecir si mañana el precio cerrará más alto que hoy.

Decisiones a tomar:
1. ¿Qué comparar? ¿Close(t) vs Close(t+1)? ¿Open(t) vs Close(t)?
2. ¿Clasificación binaria o multi-clase?
3. ¿Cómo evitar data leakage?

### Análisis

Día siguiente (Close t vs Close t+1)

Ventajas:
- Sin data leakage: Usamos features de t para predecir t+1
- Realista: Al cierre de hoy (4:00 PM), decidimos si comprar para vender mañana
- Actionable: La señal es ejecutable (comprar al cierre de hoy, vender al cierre de mañana)

### Análisis de Balance de Clases

Resultado obtenido:

| Clase | Frecuencia | Porcentaje |
|-------|-----------|-----------|
| Y=1 (Comprar) | 13,285 | 52.8% |
| Y=0 (Vender) | 11,875 | 47.2% |
| Total | 25,160 | 100% |

Interpretación:
- Balance casi perfecto (50/50)
- Esto indica que el mercado tiene una leve tendencia alcista (52.8% de días suben)
- No es necesario balanceo artificial (SMOTE, undersampling, class_weight)
- El modelo NO estará sesgado hacia ninguna clase

### Justificación de Negocio

En mercados financieros, una estrategia con 52.8% de acierto (suponiendo que el modelo prediga perfectamente) ya es rentable si se gestiona el riesgo correctamente. El objetivo del modelo no es predecir con 100% de precisión, sino tener una ventaja estadística (edge) sobre el 50% (aleatorio).

## Indicadores Técnicos Calculados

### Grupo 1: Indicadores de Tendencia

| Indicador | Fórmula | Parámetros | Interpretación |
|-----------|---------|------------|----------------|
| SMA_20 | Media móvil simple 20 días | window=20 | Tendencia de corto plazo |
| SMA_50 | Media móvil simple 50 días | window=50 | Tendencia de mediano plazo |
| EMA_12 | Media móvil exponencial 12 días | window=12 | Tendencia de corto plazo (más reactiva) |
| EMA_26 | Media móvil exponencial 26 días | window=26 | Tendencia de mediano plazo (más reactiva) |

Uso en trading:
- Golden Cross: SMA_20 cruza por encima de SMA_50 → Señal alcista
- Death Cross: SMA_20 cruza por debajo de SMA_50 → Señal bajista
- Precio sobre SMA: Close > SMA_50 → Tendencia alcista

Por qué incluir tanto SMA como EMA:
- SMA da igual peso a todos los días en la ventana → Más suave, menos ruido
- EMA da más peso a días recientes → Más reactiva, detecta cambios rápido
- Modelos de ML pueden aprender cuál usar según el contexto

### Grupo 2: Indicadores de Momentum

| Indicador | Fórmula | Parámetros | Interpretación |
|-----------|---------|------------|----------------|
| RSI_14 | Relative Strength Index | window=14 | Sobrecompra/sobreventa (0-100) |
| MACD | EMA(12) - EMA(26) | fast=12, slow=26 | Fuerza de tendencia |
| MACD_signal | EMA(MACD, 9) | signal=9 | Línea de señal |
| MACD_diff | MACD - MACD_signal | - | Histograma (divergencia) |

Uso en trading:
- RSI > 70: Sobrecompra → Posible corrección bajista
- RSI < 30: Sobreventa → Posible rebote alcista
- MACD cruza MACD_signal (arriba): Señal de compra
- MACD cruza MACD_signal (abajo): Señal de venta

### Grupo 3: Indicadores de Volatilidad

| Indicador | Fórmula | Parámetros | Interpretación |
|-----------|---------|------------|----------------|
| BB_upper | SMA(20) + 2×STD(20) | window=20, std=2 | Banda superior Bollinger |
| BB_middle | SMA(20) | window=20 | Banda media Bollinger |
| BB_lower | SMA(20) - 2×STD(20) | window=20, std=2 | Banda inferior Bollinger |
| BB_width | (BB_upper - BB_lower) / BB_middle | - | Ancho relativo de bandas |
| ATR_14 | Average True Range | window=14 | Volatilidad absoluta |

Uso en trading:
- Precio toca BB_upper: Sobrecompra (alta probabilidad de corrección)
- Precio toca BB_lower: Sobreventa (alta probabilidad de rebote)
- BB_width alta: Alta volatilidad → Mayor riesgo/recompensa
- BB_width baja: Baja volatilidad → Posible breakout próximo

### Grupo 4: Indicadores de Volumen

| Indicador | Fórmula | Parámetros | Interpretación |
|-----------|---------|------------|----------------|
| OBV | On-Balance Volume (acumulativo) | - | Presión compradora/vendedora |

Uso en trading:
- OBV subiendo + Precio subiendo: Confirmación de tendencia alcista
- OBV bajando + Precio subiendo: Divergencia → Posible reversión
- OBV mide volumen acumulado: Un aumento fuerte indica interés institucional

## Features Adicionales Calculadas

Además de los indicadores técnicos estándar, se calcularon 3 features adicionales:

| Feature | Fórmula | Interpretación |
|---------|---------|----------------|
| Returns | (Close(t) - Close(t-1)) / Close(t-1) | Retorno diario porcentual |
| Volatility_10 | std(Returns, window=10) | Volatilidad de corto plazo (10 días) |
| Volume_change | (Volume(t) - Volume(t-1)) / Volume(t-1) | Cambio porcentual en volumen |

### Justificación de Features Adicionales

Returns:
- Captura la dirección y magnitud del movimiento diario
- Normalizado por precio (es porcentual, no absoluto)
- Feature más importante en muchos modelos financieros

Volatility_10:
- Mide cuánto varían los retornos en los últimos 10 días
- Alta volatilidad → Mayor incertidumbre → Mayor riesgo
- Útil para estrategias de gestión de riesgo

Volume_change:
- Detecta cambios en el interés de los inversores
- Aumento de volumen + subida de precio → Tendencia fuerte
- Aumento de volumen + bajada de precio → Pánico vendedor

## Archivos Generados

### Estructura de Output

data/processed/stock_features.parquet

data/processed/ml_ready/features_combined.parquet

### Estructura de Columnas (features_combined.parquet)

| # | Columna | Tipo | Uso |
|---|---------|------|-----|
| 1 | Ticker | string | Identificador (NO usar en modelo) |
| 2 | Date | datetime | Índice temporal (NO usar en modelo) |
| 3-19 | Features (X) | float | 17 indicadores técnicos |
| 20 | Target | int | Variable Y (0=Vender, 1=Comprar) |

## Metadata Generada

### Archivo de Metadata

Ubicación: `data/metadata/transformation_log.yaml`

Contenido generado:

```yaml
transformation_date: '2025-10-25T18:15:32.123456Z'
config:
  date_range:
    start: '2015-01-01'
    end: '2024-12-31'
  indicators:
    trend: ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']
    momentum: ['RSI_14', 'MACD', 'MACD_signal', 'MACD_diff']
    volatility: ['BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'ATR_14']
    volume: ['OBV']
  additional_features: ['Returns', 'Volatility_10', 'Volume_change']
  target_definition: 'Y = 1 if Close(t+1) > Close(t)'
results:
  total_tickers: 10
  total_rows: 25160
  rows_per_ticker: 2516
  features_count: 17
  target_balance:
    buy: 0.528
    sell: 0.472
  files_generated: 11
status: success
```
---

Última revisión: 2025-10-25
Autor: Grupo 27 - Universidad de Los Andes
Proyecto: Sistema de Clasificación de Acciones S&P 500 (Análisis Técnico)
