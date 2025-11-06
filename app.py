# app.py
# Dash básico: Dropdown de tickers, rango de fechas y 4 gráficos análisis técnicos
# leyendo los .parquet generados por los scripts del modelo.

import os, re, glob
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import traceback
from dash import Dash, dcc, html, Input, Output

# === CONFIG ===
DATA_DIR = "D:/Documents/jcbl/MIAD 2025 ciclo 4/Curso Gerencia de Proyectos/Despliegue-Soluciones-Proyecto/data/processed"          # carpeta con *_features.parquet
FILE_PATTERN = "*_features.parquet"
DATE_COL = "Date"
PRICE_COLS = ["Adj Close", "Close"]

# Por defecto mostrar los últimos 3 meses (aprox. 90 días)
DEFAULT_END = date.today()
DEFAULT_START = DEFAULT_END - timedelta(days=90)

# === utilidades ===
def list_tickers():
    files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
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
    # fallback si no existe ninguno
    return None

def load_df(ticker):
    fpath = os.path.join(DATA_DIR, f"{ticker}_features.parquet")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No existe: {fpath}")
    df = pd.read_parquet(fpath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    return df

def slice_by_dates(df, start_date, end_date):
    """Filtra el DataFrame por rango de fechas.

    Acepta start_date/end_date como strings (ISO) o None. Si falta alguno, usa límite del DF.
    Esto evita problemas por formatos/hora y hace la comparación inclusiva.
    """
    # Asegurar que Date sea datetime
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Obtener timezone de la columna (si existe)
    col_tz = None
    try:
        col_tz = df[DATE_COL].dt.tz
    except Exception:
        col_tz = None

    # Construir timestamps de inicio/fin respetando zona horaria de la columna
    if start_date:
        start_ts = pd.to_datetime(start_date)
    else:
        start_ts = df[DATE_COL].min()

    if end_date:
        end_ts = pd.to_datetime(end_date)
    else:
        end_ts = df[DATE_COL].max()

    # Normalizar (quitar hora) y ajustar timezone si la columna tiene tz
    # Normalizar mantiene la zona si el timestamp la tiene
    try:
        # Si la columna tiene tz y los límites no, localizarlos
        if col_tz is not None:
            if pd.api.types.is_datetime64tz_dtype(df[DATE_COL]):
                # start_ts
                if getattr(start_ts, 'tzinfo', None) is None:
                    start_ts = pd.to_datetime(start_ts).tz_localize(col_tz)
                else:
                    start_ts = pd.to_datetime(start_ts).tz_convert(col_tz)
                if getattr(end_ts, 'tzinfo', None) is None:
                    end_ts = pd.to_datetime(end_ts).tz_localize(col_tz)
                else:
                    end_ts = pd.to_datetime(end_ts).tz_convert(col_tz)
        else:
            # columna sin tz -> normalizar a naive
            start_ts = pd.to_datetime(start_ts)
            end_ts = pd.to_datetime(end_ts)
    except Exception:
        # En caso de cualquier problema con tz, convertir columna a naive
        df[DATE_COL] = df[DATE_COL].dt.tz_convert(None) if pd.api.types.is_datetime64tz_dtype(df[DATE_COL]) else df[DATE_COL]
        start_ts = pd.to_datetime(start_ts)
        end_ts = pd.to_datetime(end_ts)

    # Normalizar a inicio de día (manteniendo tz si aplica)
    try:
        start_ts = start_ts.normalize()
        end_ts = end_ts.normalize()
    except Exception:
        # fallback: si normalize falla, convertir sin normalize
        start_ts = pd.to_datetime(start_ts)
        end_ts = pd.to_datetime(end_ts)

    mask = (df[DATE_COL] >= start_ts) & (df[DATE_COL] <= end_ts)
    return df.loc[mask].copy()

# === figuras ===
def fig_moving_averages(df):
    price_col = pick_price_col(df.columns)
    fig = go.Figure()
    if price_col:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[price_col], mode="lines", name=price_col))
    # MAs exactas que dijiste
    for col in ["SMA_20", "SMA_50", "EMA_12", "EMA_26"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col], mode="lines", name=col))
    fig.update_layout(title="Medias móviles (precio + SMA_20, SMA_50, EMA_12, EMA_26)",
                      legend=dict(orientation="h"), margin=dict(l=10,r=10,t=40,b=10))
    return fig

def fig_rsi(df):
    if "RSI_14" not in df.columns:
        return go.Figure(layout=dict(title="RSI_14 no encontrado", margin=dict(l=10,r=10,t=40,b=10)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[DATE_COL], y=df["RSI_14"], mode="lines", name="RSI_14"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI (14)", yaxis=dict(range=[0,100]),
                      legend=dict(orientation="h"), margin=dict(l=10,r=10,t=40,b=10))
    return fig

def fig_macd(df):
    # MACD, señal y histograma (diff)
    fig = go.Figure()
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df["MACD"], mode="lines", name="MACD"))
    if "MACD_signal" in df.columns:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df["MACD_signal"], mode="lines", name="MACD_signal"))
    if "MACD_diff" in df.columns:
        fig.add_trace(go.Bar(x=df[DATE_COL], y=df["MACD_diff"], name="MACD_diff", opacity=0.35))
    fig.update_layout(title="MACD (línea, señal e histograma)",
                      legend=dict(orientation="h"), margin=dict(l=10,r=10,t=40,b=10))
    return fig

def fig_bbands(df):
    price_col = pick_price_col(df.columns)
    fig = go.Figure()
    if price_col:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[price_col], mode="lines", name=price_col))
    for col in ["BB_upper", "BB_middle", "BB_lower"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col], mode="lines", name=col))
    fig.update_layout(title="Bandas de Bollinger (precio + upper/middle/lower)",
                      legend=dict(orientation="h"), margin=dict(l=10,r=10,t=40,b=10))
    return fig

# === app Dash ===
TICKERS = list_tickers()
if not TICKERS:
    raise SystemExit(f"No se encontraron archivos {FILE_PATTERN} en {DATA_DIR}")

# Determinar rango de fechas global disponible (min, max) leyendo los archivos
try:
    all_mins = []
    all_maxs = []
    for t in TICKERS:
        try:
            _df = load_df(t)
            if not _df.empty:
                all_mins.append(_df[DATE_COL].min().date())
                all_maxs.append(_df[DATE_COL].max().date())
        except Exception:
            # si algún ticker falla, lo saltamos
            continue
    if all_maxs:
        GLOBAL_MAX_DATE = max(all_maxs)
    else:
        GLOBAL_MAX_DATE = date.today()
    if all_mins:
        GLOBAL_MIN_DATE = min(all_mins)
    else:
        GLOBAL_MIN_DATE = date(2010,1,1)
except Exception:
    GLOBAL_MAX_DATE = date.today()
    GLOBAL_MIN_DATE = date(2010,1,1)

# Valores por defecto: últimos 90 días desde la última fecha disponible
DEFAULT_END = GLOBAL_MAX_DATE
DEFAULT_START = DEFAULT_END - timedelta(days=90)

app = Dash(__name__)
app.title = "Analítica Técnica – Demo Fase 1"

app.layout = html.Div([
    html.H2("Modelo de recomendación – Análisis técnico (Fase 1)"),
    html.Div([
        html.Div([
            html.Label("Ticker"),
            dcc.Dropdown(options=TICKERS, value=TICKERS[0], id="dd-ticker", clearable=False),
            html.Br(),
            html.Label("Rango de fechas"),
            dcc.DatePickerRange(
                id="dp-range",
                start_date=DEFAULT_START,
                end_date=DEFAULT_END,
                min_date_allowed=GLOBAL_MIN_DATE,
                max_date_allowed=DEFAULT_END
            ),
            html.Br(), html.Br(),
            html.Button("Actualizar", id="btn-update", n_clicks=0),
        ], style={"width":"20%","display":"inline-block","verticalAlign":"top","padding":"8px","borderRight":"1px solid #ddd"}),

        html.Div([
            html.H3("Análisis Técnico"),
            # Primera fila: MA y RSI
            html.Div([
                html.Div(dcc.Loading(dcc.Graph(id="g-ma")), style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div(dcc.Loading(dcc.Graph(id="g-rsi")), style={"width": "49%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "2%"}),
            ], style={"width": "100%"}),
            # Segunda fila: MACD y BBands
            html.Div([
                html.Div(dcc.Loading(dcc.Graph(id="g-macd")), style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div(dcc.Loading(dcc.Graph(id="g-bbands")), style={"width": "49%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "2%"}),
            ], style={"width": "100%", "marginTop": "8px"}),
        ], style={"width":"78%","display":"inline-block","padding":"8px"})
    ])
], style={"fontFamily":"Calibri, Arial, sans-serif"})

@app.callback(
    Output("g-ma","figure"),
    Output("g-rsi","figure"),
    Output("g-macd","figure"),
    Output("g-bbands","figure"),
    Input("btn-update","n_clicks"),
    Input("dd-ticker","value"),
    Input("dp-range","start_date"),
    Input("dp-range","end_date"),
    prevent_initial_call=False
)
def update_plots(n_clicks, ticker, start_date, end_date):
    # Depuración: imprimir lo que llega desde el frontend
    try:
        print(f"update_plots called: ticker={ticker}, start_date={start_date}, end_date={end_date}, n_clicks={n_clicks}")
    except Exception:
        pass

    try:
        df = load_df(ticker)
        before = len(df)
        df = slice_by_dates(df, start_date, end_date)
        after = len(df)
        print(f"Rows before filter: {before}, after filter: {after}")

        return fig_moving_averages(df), fig_rsi(df), fig_macd(df), fig_bbands(df)

    except Exception as e:
        # Imprimir traceback completo en la consola para depuración
        tb = traceback.format_exc()
        print("Exception in update_plots:\n", tb)

        # Crear una figura de error simple para mostrar en cada gráfico
        def error_figure(msg):
            f = go.Figure()
            f.update_layout(title=f"Error: {msg}", margin=dict(l=10, r=10, t=40, b=10))
            # Añadir texto en el centro
            f.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                             font=dict(color="red", size=12))
            return f

        short_msg = str(e)
        # Devolver la misma figura de error para los 4 gráficos
        err_fig = error_figure(short_msg)
        return err_fig, err_fig, err_fig, err_fig

if __name__ == "__main__":
    # Ejecuta la app (http://127.0.0.1:8050 por defecto)
    # En versiones recientes de Dash se usa app.run(...) en lugar de app.run_server(...)
    app.run(debug=True)
