# ===============================================================
#  APP.PY – Dashboard Completo (Fase 1 + Fase 2)
#  - Usa datos locales (ETL) para gráficos y backtests
#  - Usa API /api/v1/predict para la recomendación final
# ===============================================================

import os, re, glob
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import traceback
import requests

# ===============================================================
# CONFIGURACIÓN GENERAL
# ===============================================================

API_BASE = "http://44.211.229.94:8001"

# Fase 1 (Análisis técnico)
DATA_PROCESSED = "data/processed"
FILE_PATTERN = "*_features.parquet"
DATE_COL = "Date"
PRICE_COLS = ["Adj Close", "Close"]

# Fase 2 (Dashboard del modelo)
DASH_DIR = "data/dashboard"
PRICES_DIR = f"{DASH_DIR}/prices"
VALIDATION_DIR = f"{DASH_DIR}/validation"
RESULTS_DIR = "data/model_results"

# Límite superior de fechas según ETL
LAST_DATA_DATE = date(2024, 12, 31)

# Rango por defecto (últimos 90 días hasta LAST_DATA_DATE)
DEFAULT_END = LAST_DATA_DATE
DEFAULT_START = DEFAULT_END - timedelta(days=90)

# Estilo de descripciones cortas bajo cada gráfico/tabla
DESC_STYLE = {
    "fontSize": "16px",
    "color": "#444",
    "marginTop": "6px",
    "fontWeight": "600"
}

# ===============================================================
# UTILIDADES FECHAS
# ===============================================================

def normalize_datetime_series(s):
    """
    Convierte una serie a datetime, forzando a UTC y eliminando la zona horaria.
    """
    s = pd.to_datetime(s, utc=True, errors="coerce")
    s = s.dt.tz_convert(None)
    return s

# ===============================================================
# UTILIDADES FASE 1
# ===============================================================

def list_tickers():
    files = glob.glob(os.path.join(DATA_PROCESSED, FILE_PATTERN))
    tickers = []
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"([A-Za-z\.^-]+)_features\.parquet", base)
        if m:
            tickers.append(m.group(1))
    return sorted(tickers)

def pick_price_col(cols):
    for c in PRICE_COLS:
        if c in cols:
            return c
    return None

def load_df_fase1(ticker):
    fpath = os.path.join(DATA_PROCESSED, f"{ticker}_features.parquet")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No existe: {fpath}")
    df = pd.read_parquet(fpath)
    df[DATE_COL] = normalize_datetime_series(df[DATE_COL])
    return df.sort_values(DATE_COL)

def slice_by_dates(df, start_date, end_date):
    df = df.copy()
    df[DATE_COL] = normalize_datetime_series(df[DATE_COL])

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if getattr(start_ts, "tzinfo", None) is not None:
        start_ts = start_ts.tz_localize(None)
    if getattr(end_ts, "tzinfo", None) is not None:
        end_ts = end_ts.tz_localize(None)

    start_ts = start_ts.normalize()
    end_ts = end_ts.normalize()

    mask = (df[DATE_COL] >= start_ts) & (df[DATE_COL] <= end_ts)
    return df.loc[mask].copy()

# ===============================================================
# GRÁFICOS FASE 1
# ===============================================================

def fig_moving_averages(df):
    price_col = pick_price_col(df.columns)
    fig = go.Figure()
    if price_col:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[price_col],
                                 mode="lines", name=price_col))
    for col in ["SMA_20", "SMA_50", "EMA_12", "EMA_26"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col],
                                     mode="lines", name=col))
    fig.update_layout(title="Medias Móviles", legend=dict(orientation="h"))
    return fig

def fig_rsi(df):
    fig = go.Figure()
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df["RSI_14"],
                                 mode="lines", name="RSI_14"))
        fig.add_hline(y=70, line_dash="dash")
        fig.add_hline(y=30, line_dash="dash")
    else:
        fig.add_annotation(text="RSI_14 no encontrado",
                           x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title="RSI (14)", yaxis=dict(range=[0, 100]))
    return fig

def fig_macd(df):
    fig = go.Figure()
    for col in ["MACD", "MACD_signal"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col],
                                     mode="lines", name=col))
    if "MACD_diff" in df.columns:
        fig.add_trace(go.Bar(x=df[DATE_COL], y=df["MACD_diff"],
                             name="MACD_diff", opacity=0.35))
    fig.update_layout(title="MACD")
    return fig

def fig_bbands(df):
    price_col = pick_price_col(df.columns)
    fig = go.Figure()
    if price_col:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[price_col],
                                 mode="lines", name=price_col))
    for col in ["BB_upper", "BB_middle", "BB_lower"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col],
                                     mode="lines", name=col))
    fig.update_layout(title="Bandas de Bollinger")
    return fig

# ===============================================================
# UTILIDADES FASE 2 – DATOS LOCALES
# ===============================================================

def load_prices(ticker):
    """
    Precios del ticker:
    1) Busca en data/dashboard/prices/{ticker}_prices.parquet
    2) Si no existe, busca en data/raw_parquet con prefijo del ticker.
    """
    path = os.path.join(PRICES_DIR, f"{ticker}_prices.parquet")
    if os.path.exists(path):
        print(f"[load_prices] Usando {path} para {ticker}")
        df = pd.read_parquet(path)
    else:
        raw_dir = "data/raw_parquet"
        if not os.path.isdir(raw_dir):
            print(f"[load_prices] No existe {raw_dir}")
            return pd.DataFrame()

        candidates = [
            f for f in os.listdir(raw_dir)
            if f.lower().startswith(ticker.lower()) and f.lower().endswith(".parquet")
        ]
        if not candidates:
            print(f"[load_prices] No se encontró parquet para {ticker} en {raw_dir}")
            return pd.DataFrame()

        raw_path = os.path.join(raw_dir, candidates[0])
        print(f"[load_prices] Usando {raw_path} para {ticker}")
        df = pd.read_parquet(raw_path)

    if "date" in df.columns:
        df["date"] = normalize_datetime_series(df["date"])
    elif "Date" in df.columns:
        df["date"] = normalize_datetime_series(df["Date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        df["date"] = normalize_datetime_series(df.index.to_series())
    else:
        print(f"[load_prices] No se encontró columna de fecha en parquet de {ticker}")
        return pd.DataFrame()

    return df

def load_sp500():
    """
    Carga el S&P500:
    1) data/dashboard/prices/SP500_prices.parquet
    2) Si no existe, lo construye desde data/raw_parquet
       promediando los cierres de todos los tickers.
    """
    path_prices = f"{PRICES_DIR}/SP500_prices.parquet"
    if os.path.exists(path_prices):
        df = pd.read_parquet(path_prices)
        if "date" in df.columns:
            df["date"] = normalize_datetime_series(df["date"])
        elif "Date" in df.columns:
            df["date"] = normalize_datetime_series(df["Date"])
        return df

    raw_dir = "data/raw_parquet"
    if not os.path.isdir(raw_dir):
        print("Directorio data/raw_parquet no existe para S&P500.")
        return pd.DataFrame()

    frames = []
    close_candidates = ["Adj Close", "Close", "adj_close", "close"]

    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".parquet"):
            continue
        fpath = os.path.join(raw_dir, fname)
        try:
            df = pd.read_parquet(fpath)

            if "Date" in df.columns:
                date_series = normalize_datetime_series(df["Date"])
            elif "date" in df.columns:
                date_series = normalize_datetime_series(df["date"])
            elif isinstance(df.index, pd.DatetimeIndex):
                date_series = normalize_datetime_series(df.index.to_series())
            else:
                continue

            price_col = None
            for c in close_candidates:
                if c in df.columns:
                    price_col = c
                    break
            if price_col is None:
                continue

            tmp = pd.DataFrame({
                "date": date_series,
                "close": df[price_col].astype(float)
            })
            frames.append(tmp)
        except Exception as e:
            print(f"Error leyendo {fpath} para S&P500: {e}")
            continue

    if not frames:
        print("No se pudo construir S&P500 desde data/raw_parquet.")
        return pd.DataFrame()

    all_data = pd.concat(frames, ignore_index=True)
    agg = all_data.groupby("date", as_index=False)["close"].mean().sort_values("date")
    print("S&P500 agregado construido desde data/raw_parquet.")
    return agg

def load_backtest(ticker):
    f = f"{VALIDATION_DIR}/backtest_{ticker}.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        df["date"] = normalize_datetime_series(df["date"])
        return df
    return pd.DataFrame()

def load_backtest_metrics(ticker):
    f = f"{VALIDATION_DIR}/metrics_{ticker}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)
    return pd.DataFrame()

def load_corr():
    """
    Matriz de correlación desde data/model_results/features_corr.csv.
    """
    csv_path = os.path.join(RESULTS_DIR, "features_corr.csv")
    if os.path.exists(csv_path):
        try:
            corr = pd.read_csv(csv_path, index_col=0)
            print(f"[load_corr] Matriz de correlación {corr.shape} desde {csv_path}")
            return corr
        except Exception as e:
            print(f"[load_corr] Error leyendo {csv_path}: {e}")
            return pd.DataFrame()
    else:
        print(f"[load_corr] No se encontró {csv_path}")
        return pd.DataFrame()

def load_roc():
    f = os.path.join(RESULTS_DIR, "roc_curve.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    return pd.DataFrame()

def load_classification_metrics():
    f = os.path.join(RESULTS_DIR, "classification_metrics.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    return pd.DataFrame()

# ===============================================================
# UTILIDAD PRECIO MERCADO
# ===============================================================

def pick_market_price_col(df):
    for c in ["close", "Close", "Adj Close", "adj_close", "Adj_Close"]:
        if c in df.columns:
            return c
    return None

# ===============================================================
# UTILIDADES – LLAMADA A LA API DE PREDICCIÓN
# ===============================================================

API_FEATURE_COLUMNS = {
    "SMA_20": "SMA_20",
    "SMA_50": "SMA_50",
    "EMA_12": "EMA_12",
    "EMA_26": "EMA_26",
    "RSI_14": "RSI_14",
    "MACD": "MACD",
    "MACD_signal": "MACD_signal",
    "MACD_diff": "MACD_diff",
    "BB_upper": "BB_upper",
    "BB_middle": "BB_middle",
    "BB_lower": "BB_lower",
    "BB_width": "BB_width",
    "ATR_14": "ATR_14",
    "OBV": "OBV",
    "Retornos": "Returns",
    "Volatility_10": "Volatility_10",
    "Volume_change": "Volume_change",
}

def build_api_inputs_from_df(df_features):
    records = []
    for _, row in df_features.iterrows():
        item = {}
        for api_name, df_col in API_FEATURE_COLUMNS.items():
            if df_col in df_features.columns:
                val = row[df_col]
                if pd.notna(val):
                    item[api_name] = float(val)
        if item:
            records.append(item)
    return {"inputs": records}

def get_prediction_from_api(df_features):
    if df_features.empty:
        return None, None

    df_last = df_features.tail(1)
    payload = build_api_inputs_from_df(df_last)

    if not payload.get("inputs"):
        print("Payload sin 'inputs' (revisar API_FEATURE_COLUMNS).")
        return None, None

    try:
        url = f"{API_BASE}/api/v1/predict"
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code != 200:
            print("Error API /predict:", resp.status_code, resp.text)
            resp.raise_for_status()

        data = resp.json()
        preds = data.get("predicciones") or data.get("predictions")
        probs = data.get("probabilidades") or data.get("probabilities")

        if not preds:
            print("La API no devolvió 'predicciones':", data)
            return None, None

        pred = preds[-1]
        proba = probs[-1] if probs else None
        return pred, proba

    except Exception as e:
        print("Error llamando a la API de predicción:", e)
        return None, None

# ===============================================================
# LAYOUT
# ===============================================================

TICKERS = list_tickers()
print("TICKERS encontrados:", TICKERS)

app = Dash(__name__)
app.title = "Dashboard – Modelo de Recomendación"

if not TICKERS:
    app.layout = html.Div(
        [
            html.H2("Dashboard – Modelo de Recomendación"),
            html.P("No se encontraron archivos *_features.parquet en data/processed."),
            html.P("Verifica que hayas ejecutado el ETL y que existan esos archivos.")
        ],
        style={"padding": "40px", "fontFamily": "Calibri, Arial"}
    )
else:
    app.layout = html.Div([
        html.H2("Modelo de recomendación – Fase 1 + Fase 2"),

        html.Div([
            # Columna izquierda: controles + recomendación
            html.Div([
                html.Label("Ticker"),
                dcc.Dropdown(TICKERS, value=TICKERS[0],
                             id="dd-ticker", clearable=False),

                html.Br(),
                html.Label("Rango de Fechas"),
                dcc.DatePickerRange(
                    id="dp-range",
                    start_date=DEFAULT_START,
                    end_date=DEFAULT_END,
                    max_date_allowed=LAST_DATA_DATE
                ),

                html.Br(), html.Br(),
                html.Button("Actualizar", id="btn-update", n_clicks=0),

                html.Br(), html.Br(),
                html.H4("Recomendación del Modelo (API)"),
                html.Div(
                    id="recommendation-card",
                    style={
                        "padding": "15px",
                        "border": "1px solid #aaa",
                        "borderRadius": "8px",
                        "background": "#f5f5f5",
                        "marginTop": "10px"
                    }
                )
            ], style={
                "width": "24%",
                "display": "inline-block",
                "verticalAlign": "top",
                "padding": "8px",
                "borderRight": "1px solid #aaa"
            }),

            # Columna derecha: análisis técnico y resultados
            html.Div([
                html.H3("Fase 1 – Análisis Técnico"),

                # Fila 1: MA y RSI
                html.Div([
                    html.Div([
                        dcc.Graph(id="g-ma"),
                        html.P(
                            "Precio y medias móviles del activo.",
                            style=DESC_STYLE
                        ),
                    ], style={"width": "49%", "display": "inline-block"}),

                    html.Div([
                        dcc.Graph(id="g-rsi"),
                        html.P(
                            "RSI para detectar sobrecompra y sobreventa.",
                            style=DESC_STYLE
                        ),
                    ], style={"width": "49%", "display": "inline-block"}),
                ]),

                # Fila 2: MACD y Bandas
                html.Div([
                    html.Div([
                        dcc.Graph(id="g-macd"),
                        html.P(
                            "MACD y señal para medir el impulso.",
                            style=DESC_STYLE
                        ),
                    ], style={"width": "49%", "display": "inline-block"}),

                    html.Div([
                        dcc.Graph(id="g-bbands"),
                        html.P(
                            "Bandas de Bollinger y precio de cierre.",
                            style=DESC_STYLE
                        ),
                    ], style={"width": "49%", "display": "inline-block"}),
                ], style={"marginTop": "10px"}),

                html.H3("Fase 2 – Desempeño del Modelo", style={"marginTop": "40px"}),

                # Fila 3: Backtesting y Mercado
                html.Div([
                    html.Div([
                        html.H4("Backtesting del Modelo"),
                        dcc.Graph(id="g-backtest"),
                        html.Div(id="metrics-backtest"),
                        html.P(
                            "Estrategia vs comprar y mantener.",
                            style=DESC_STYLE
                        ),
                    ], style={"width": "49%", "display": "inline-block"}),

                    html.Div([
                        html.H4("Información del Mercado"),
                        dcc.Graph(id="g-market-info"),
                        html.P(
                            "Serie del S&P 500 y del ticker.",
                            style=DESC_STYLE
                        ),
                    ], style={"width": "49%", "display": "inline-block"}),
                ], style={"marginTop": "10px"}),

                # Fila 4: Resultados del modelo y ROC apilados
                html.Div([
                    html.H4("Resultados del modelo"),
                    dcc.Graph(id="g-corr"),
                    html.Div(id="cls-metrics"),
                    html.P(
                        "Matriz de correlación y métricas del modelo.",
                        style=DESC_STYLE
                    ),

                    html.H4("Curva ROC", style={"marginTop": "20px"}),
                    dcc.Graph(id="g-roc"),
                    html.P(
                        "Curva ROC para el ticker seleccionado.",
                        style=DESC_STYLE
                    ),
                ], style={"marginTop": "10px"}),

            ], style={"width": "74%", "display": "inline-block", "padding": "15px"})
        ])
    ])

# ===============================================================
# CALLBACK FASE 1 + FASE 2
# ===============================================================

@app.callback(
    Output("g-ma", "figure"),
    Output("g-rsi", "figure"),
    Output("g-macd", "figure"),
    Output("g-bbands", "figure"),
    Output("g-backtest", "figure"),
    Output("metrics-backtest", "children"),
    Output("g-market-info", "figure"),
    Output("g-corr", "figure"),
    Output("g-roc", "figure"),
    Output("cls-metrics", "children"),
    Output("recommendation-card", "children"),
    Input("btn-update", "n_clicks"),
    Input("dd-ticker", "value"),
    Input("dp-range", "start_date"),
    Input("dp-range", "end_date"),
)
def update_all(n_clicks, ticker, start_date, end_date):
    try:
        if start_date is None:
            start_date = DEFAULT_START
        if end_date is None:
            end_date = DEFAULT_END

        end_d = pd.to_datetime(end_date).date()
        if end_d > LAST_DATA_DATE:
            end_date = LAST_DATA_DATE

        # Fase 1
        df1 = load_df_fase1(ticker)
        df1 = slice_by_dates(df1, start_date, end_date)

        fig1 = fig_moving_averages(df1)
        fig2 = fig_rsi(df1)
        fig3 = fig_macd(df1)
        fig4 = fig_bbands(df1)

        # Backtesting
        bt = load_backtest(ticker)
        if not bt.empty:
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            if getattr(start_ts, "tzinfo", None) is not None:
                start_ts = start_ts.tz_localize(None)
            if getattr(end_ts, "tzinfo", None) is not None:
                end_ts = end_ts.tz_localize(None)

            bt = bt[(bt["date"] >= start_ts) & (bt["date"] <= end_ts)]

            fig_bt = go.Figure()
            if not bt.empty:
                fig_bt.add_trace(go.Scatter(x=bt["date"], y=bt["cum_strategy"],
                                            name="Estrategia"))
                fig_bt.add_trace(go.Scatter(x=bt["date"], y=bt["cum_benchmark"],
                                            name="Buy & Hold"))
                fig_bt.update_layout(xaxis_title="Fecha",
                                     yaxis_title="Valor acumulado")
            else:
                fig_bt.add_annotation(text="Sin datos de backtest en el rango",
                                      x=0.5, y=0.5, showarrow=False)
        else:
            fig_bt = go.Figure()
            fig_bt.add_annotation(text="Sin datos de backtest",
                                  x=0.5, y=0.5, showarrow=False)

        metrics_df = load_backtest_metrics(ticker)
        if not metrics_df.empty:
            cards = []
            for _, row in metrics_df.iterrows():
                val = row.get("strategy", None)
                txt_val = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
                cards.append(
                    html.Div([
                        html.H5(row.get("metric", "")),
                        html.P(txt_val)
                    ], style={
                        "display": "inline-block",
                        "padding": "8px",
                        "marginRight": "10px",
                        "border": "1px solid #ccc",
                        "borderRadius": "6px",
                        "backgroundColor": "white"
                    })
                )
            metrics_cards = cards
        else:
            metrics_cards = html.P("Sin métricas disponibles.")

        # Mercado (S&P 500 + ticker)
        df_t = load_prices(ticker)
        df_sp = load_sp500()

        fig_mkt = make_subplots(specs=[[{"secondary_y": True}]])
        any_trace = False

        if not df_sp.empty:
            df_sp_f = df_sp[(df_sp["date"] >= pd.to_datetime(start_date)) &
                            (df_sp["date"] <= pd.to_datetime(end_date))]
            sp_col = pick_market_price_col(df_sp_f)
            if sp_col and not df_sp_f.empty:
                fig_mkt.add_trace(
                    go.Scatter(x=df_sp_f["date"], y=df_sp_f[sp_col], name="S&P 500"),
                    secondary_y=False
                )
                any_trace = True

        if not df_t.empty:
            df_t_f = df_t[(df_t["date"] >= pd.to_datetime(start_date)) &
                          (df_t["date"] <= pd.to_datetime(end_date))]
            t_col = pick_market_price_col(df_t_f)
            if t_col and not df_t_f.empty:
                fig_mkt.add_trace(
                    go.Scatter(x=df_t_f["date"], y=df_t_f[t_col], name=ticker),
                    secondary_y=True
                )
                any_trace = True

        fig_mkt.update_xaxes(title_text="Fecha")
        fig_mkt.update_yaxes(title_text="S&P 500", secondary_y=False)
        fig_mkt.update_yaxes(title_text=f"{ticker} cierre", secondary_y=True)

        if not any_trace:
            fig_mkt.add_annotation(
                text="Sin datos de mercado para el rango/ticker seleccionado.",
                x=0.5, y=0.5, showarrow=False
            )

        # Matriz de correlación
        c = load_corr()
        if not c.empty:
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=c.values,
                    x=c.columns,
                    y=c.index,
                    zmin=-1,
                    zmax=1,
                    colorbar_title="corr"
                )
            )
            fig_corr.update_layout(
                margin=dict(l=60, r=20, t=30, b=40),
                height=450,
                yaxis=dict(automargin=True)
            )
        else:
            fig_corr = go.Figure()
            fig_corr.add_annotation(
                text="Sin matriz de correlación",
                x=0.5, y=0.5, showarrow=False
            )

        # Curva ROC (solo ticker seleccionado)
        r = load_roc()
        fig_roc = go.Figure()

        if not r.empty:
            if not {"ticker", "fpr", "tpr"}.issubset(set(r.columns)):
                fig_roc.add_annotation(
                    text="roc_curve.csv debe incluir: ticker, fpr, tpr.",
                    x=0.5, y=0.5, showarrow=False
                )
            else:
                df_tkr = r[r["ticker"] == ticker]

                if df_tkr.empty:
                    fig_roc.add_annotation(
                        text=f"Sin curva ROC para {ticker}.",
                        x=0.5, y=0.5, showarrow=False
                    )
                else:
                    fig_roc.add_trace(
                        go.Scatter(
                            x=df_tkr["fpr"],
                            y=df_tkr["tpr"],
                            mode="lines",
                            name=f"ROC {ticker}"
                        )
                    )
                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="azar",
                            line=dict(dash="dash")
                        )
                    )

                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=350,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
        else:
            fig_roc.add_annotation(
                text="Sin datos ROC",
                x=0.5, y=0.5, showarrow=False
            )

        # Métricas de clasificación
        cls = load_classification_metrics()
        if not cls.empty:
            if "ticker" in cls.columns:
                cls_t = cls[cls["ticker"] == ticker]
            else:
                cls_t = cls

            if not cls_t.empty:
                cols_show = [c for c in cls_t.columns if c not in ["support"]]
                table = html.Table([
                    html.Thead(html.Tr([html.Th(c) for c in cols_show])),
                    html.Tbody([
                        html.Tr([
                            html.Td(cls_t.iloc[i][c]) for c in cols_show
                        ]) for i in range(len(cls_t))
                    ])
                ], style={"fontSize": "11px", "width": "100%"})
            else:
                table = html.P("Sin métricas para este ticker.")
        else:
            table = html.P("Sin métricas de clasificación.")

        # Recomendación final (API)
        if not df1.empty:
            pred, proba = get_prediction_from_api(df1)

            if pred is not None:
                if pred == 1:
                    rec = "COMPRAR / MANTENER"
                elif pred == 0:
                    rec = "NO POSICIÓN"
                else:
                    rec = "VENDER / CERRAR POSICIÓN"

                last_date = df1[DATE_COL].max().date()
                rec_text = f"Para {last_date} el modelo recomienda: {rec}"
                if proba is not None:
                    rec_text += f" (probabilidad de subida: {float(proba):.2%})"
            else:
                rec_text = "No se pudo obtener una recomendación desde la API."
        else:
            rec_text = "Sin datos de features en el rango seleccionado."

        rec_card = html.Div([
            html.P(rec_text),
            html.Ul([
                html.Li("La recomendación se calcula con el modelo entrenado."),
                html.Li("Utilizar como apoyo y no como única fuente de decisión.")
            ], style={"fontSize": "11px", "color": "#555"})
        ])

        return (
            fig1, fig2, fig3, fig4,
            fig_bt, metrics_cards, fig_mkt, fig_corr, fig_roc,
            table, rec_card
        )

    except Exception as e:
        print("ERROR en update_all:\n", traceback.format_exc())
        empty_fig = go.Figure()
        err_msg = html.P(f"Error en el callback: {e}")
        return (
            empty_fig, empty_fig, empty_fig, empty_fig,
            empty_fig, err_msg,
            empty_fig, empty_fig, empty_fig,
            err_msg, err_msg
        )

if __name__ == "__main__":
    print("Iniciando servidor Dash...")
    try:
        app.run(debug=True)
    except Exception as e:
        print("Error al iniciar la app:", e)
        traceback.print_exc()

