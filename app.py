# ===============================================================
#  APP.PY – Dashboard Completo (Fase 1 + Fase 2 Mockup)
#  Reemplaza por completo el archivo anterior
#  Usa datos desde: data/dashboard/
# ===============================================================

import os, re, glob
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import traceback

# ===============================================================
# CONFIGURACIÓN GENERAL
# ===============================================================

# FASE 1 (Análisis Técnico)
DATA_PROCESSED = "data/processed"
FILE_PATTERN = "*_features.parquet"
DATE_COL = "Date"
PRICE_COLS = ["Adj Close", "Close"]

# FASE 2 (Dashboard del Mockup)
DASH_DIR = "data/dashboard"
PRICES_DIR = f"{DASH_DIR}/prices"
VALIDATION_DIR = f"{DASH_DIR}/validation"
RESULTS_DIR = f"{DASH_DIR}/model_results"

DEFAULT_END = date.today()
DEFAULT_START = DEFAULT_END - timedelta(days=90)

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
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)
    return df

def slice_by_dates(df, start_date, end_date):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Si la columna tiene timezone (ej. America/New_York), la volvemos naive
    if pd.api.types.is_datetime64tz_dtype(df[DATE_COL]):
        df[DATE_COL] = df[DATE_COL].dt.tz_convert(None)

    # Construimos límites
    if start_date:
        start_ts = pd.to_datetime(start_date)
    else:
        start_ts = df[DATE_COL].min()
    if end_date:
        end_ts = pd.to_datetime(end_date)
    else:
        end_ts = df[DATE_COL].max()

    # Aseguramos que también sean naive
    if getattr(start_ts, "tzinfo", None) is not None:
        # si llega con tz, la quitamos
        start_ts = start_ts.tz_convert(None) if hasattr(start_ts, "tz_convert") else start_ts.tz_localize(None)
    if getattr(end_ts, "tzinfo", None) is not None:
        end_ts = end_ts.tz_convert(None) if hasattr(end_ts, "tz_convert") else end_ts.tz_localize(None)

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
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[price_col], mode="lines", name=price_col))
    for col in ["SMA_20", "SMA_50", "EMA_12", "EMA_26"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col], mode="lines", name=col))
    fig.update_layout(title="Medias Móviles", legend=dict(orientation="h"))
    return fig

def fig_rsi(df):
    fig = go.Figure()
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df["RSI_14"], mode="lines", name="RSI_14"))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    else:
        fig.add_annotation(text="RSI_14 no encontrado", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title="RSI (14)", yaxis=dict(range=[0,100]))
    return fig

def fig_macd(df):
    fig = go.Figure()
    for col in ["MACD", "MACD_signal"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col], mode="lines", name=col))
    if "MACD_diff" in df.columns:
        fig.add_trace(go.Bar(x=df[DATE_COL], y=df["MACD_diff"], name="MACD_diff", opacity=0.35))
    fig.update_layout(title="MACD")
    return fig

def fig_bbands(df):
    price_col = pick_price_col(df.columns)
    fig = go.Figure()
    if price_col:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[price_col], mode="lines", name=price_col))
    for col in ["BB_upper", "BB_middle", "BB_lower"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[col], mode="lines", name=col))
    fig.update_layout(title="Bandas de Bollinger")
    return fig

# ===============================================================
# UTILIDADES FASE 2
# ===============================================================

def load_prices(ticker):
    path = f"{PRICES_DIR}/{ticker}_prices.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_sp500():
    path = f"{PRICES_DIR}/SP500_prices.parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_backtest(ticker):
    f = f"{VALIDATION_DIR}/backtest_{ticker}.csv"
    if os.path.exists(f):
        df = pd.read_csv(f, parse_dates=["date"])
        return df
    return pd.DataFrame()

def load_backtest_metrics(ticker):
    f = f"{VALIDATION_DIR}/metrics_{ticker}.csv"
    if os.path.exists(f):
        return pd.read_csv(f)
    return pd.DataFrame()

def load_corr():
    f = f"{RESULTS_DIR}/features_corr.csv"
    if os.path.exists(f):
        return pd.read_csv(f, index_col=0)
    return pd.DataFrame()

def load_roc():
    f = f"{RESULTS_DIR}/roc_curve.csv"
    if os.path.exists(f):
        return pd.read_csv(f)
    return pd.DataFrame()

def load_classification_metrics():
    f = f"{RESULTS_DIR}/classification_metrics.csv"
    if os.path.exists(f):
        return pd.read_csv(f)
    return pd.DataFrame()

def load_signals_all():
    f = f"{VALIDATION_DIR}/signals_all_tickers.parquet"
    if os.path.exists(f):
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        # 👇 si viene con timezone (America/New_York), lo quitamos
        if pd.api.types.is_datetime64tz_dtype(df["date"]):
            df["date"] = df["date"].dt.tz_convert(None)
        return df
    return pd.DataFrame()


signals_all_cache = load_signals_all()

# ===============================================================
# LAYOUT
# ===============================================================

TICKERS = list_tickers()
print("TICKERS encontrados:", TICKERS)

app = Dash(__name__)
app.title = "Dashboard – Modelo de Recomendación"

# Si no hay tickers, muestra un mensaje simple y NO construye el layout complejo
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
            html.Div([
                html.Label("Ticker"),
                dcc.Dropdown(TICKERS, value=TICKERS[0], id="dd-ticker", clearable=False),

                html.Br(),
                html.Label("Rango de Fechas"),
                dcc.DatePickerRange(
                    id="dp-range",
                    start_date=DEFAULT_START,
                    end_date=DEFAULT_END
                ),

                html.Br(), html.Br(),
                html.Button("Actualizar", id="btn-update", n_clicks=0),

            ], style={
                "width":"20%",
                "display":"inline-block",
                "verticalAlign":"top",
                "padding":"8px",
                "borderRight":"1px solid #aaa"
            }),

            html.Div([
                html.H3("Fase 1 – Análisis Técnico"),

                html.Div([
                    html.Div(dcc.Graph(id="g-ma"), style={"width":"49%", "display":"inline-block"}),
                    html.Div(dcc.Graph(id="g-rsi"), style={"width":"49%", "display":"inline-block"}),
                ]),
                html.Div([
                    html.Div(dcc.Graph(id="g-macd"), style={"width":"49%", "display":"inline-block"}),
                    html.Div(dcc.Graph(id="g-bbands"), style={"width":"49%", "display":"inline-block"}),
                ], style={"marginTop":"10px"}),

                html.H3("Fase 2 – Dashboard del Mockup", style={"marginTop":"40px"}),

                html.Div([
                    html.Div([
                        html.H4("Backtesting del Modelo"),
                        dcc.Graph(id="g-backtest"),
                        html.Div(id="metrics-backtest")
                    ], style={"width":"50%", "display":"inline-block"}),

                    html.Div([
                        html.H4("Información del Mercado"),
                        dcc.Graph(id="g-market-info")
                    ], style={"width":"50%", "display":"inline-block"})
                ]),

                html.Div([
                    html.Div([
                        html.H4("Resultados del Modelo"),
                        dcc.Graph(id="g-corr"),
                        dcc.Graph(id="g-roc"),
                        html.Div(id="cls-metrics")
                    ], style={"width":"65%", "display":"inline-block"}),

                    html.Div([
                        html.H4("Recomendación del Modelo"),
                        html.Div(id="recommendation-card",
                                 style={"padding":"15px","border":"1px solid #aaa","borderRadius":"8px",
                                        "background":"#f5f5f5", "marginTop":"10px"})
                    ], style={"width":"33%", "display":"inline-block", "verticalAlign":"top"})
                ]),

            ], style={"width":"78%","display":"inline-block", "padding":"15px"})
        ])
    ])

# ===============================================================
# CALLBACK FASE 1 + FASE 2
# ===============================================================

@app.callback(
    Output("g-ma","figure"),
    Output("g-rsi","figure"),
    Output("g-macd","figure"),
    Output("g-bbands","figure"),

    Output("g-backtest","figure"),
    Output("metrics-backtest","children"),
    Output("g-market-info","figure"),
    Output("g-corr","figure"),
    Output("g-roc","figure"),
    Output("cls-metrics","children"),
    Output("recommendation-card","children"),

    Input("btn-update","n_clicks"),
    Input("dd-ticker","value"),
    Input("dp-range","start_date"),
    Input("dp-range","end_date"),
)
def update_all(n_clicks, ticker, start_date, end_date):
    try:
        # Normalizar fechas por si vienen como None
        if start_date is None:
            start_date = DEFAULT_START
        if end_date is None:
            end_date = DEFAULT_END

        # ================
        # FASE 1
        # ================
        df1 = load_df_fase1(ticker)
        df1 = slice_by_dates(df1, start_date, end_date)

        fig1 = fig_moving_averages(df1)
        fig2 = fig_rsi(df1)
        fig3 = fig_macd(df1)
        fig4 = fig_bbands(df1)

        # ================
        # FASE 2 - Backtesting
        # ================
        bt = load_backtest(ticker)
        if not bt.empty:
            bt = bt[(bt["date"] >= pd.to_datetime(start_date)) &
                    (bt["date"] <= pd.to_datetime(end_date))]
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt["date"], y=bt["cum_strategy"], name="Estrategia"))
            fig_bt.add_trace(go.Scatter(x=bt["date"], y=bt["cum_benchmark"], name="Buy & Hold"))
            fig_bt.update_layout(xaxis_title="Fecha", yaxis_title="Valor acumulado")
        else:
            fig_bt = go.Figure()
            fig_bt.add_annotation(text="Sin datos de backtest", x=0.5, y=0.5, showarrow=False)

        metrics_df = load_backtest_metrics(ticker)
        if not metrics_df.empty:
            cards = []
            for _, row in metrics_df.iterrows():
                val = row.get("strategy", None)
                txt_val = f"{val:.3f}" if isinstance(val, (int, float, float)) else str(val)
                cards.append(
                    html.Div([
                        html.H5(row.get("metric", "")),
                        html.P(txt_val)
                    ], style={
                        "display":"inline-block",
                        "padding":"8px",
                        "marginRight":"10px",
                        "border":"1px solid #ccc",
                        "borderRadius":"6px",
                        "backgroundColor":"white"
                    })
                )
            metrics_cards = cards
        else:
            metrics_cards = html.P("Sin métricas disponibles.")

        # ================
        # FASE 2 - Mercado (SP500 + ticker)
        # ================
        df_t = load_prices(ticker)
        df_sp = load_sp500()

        fig_mkt = make_subplots(specs=[[{"secondary_y": True}]])
        if not df_sp.empty:
            df_sp_f = df_sp[(df_sp["date"] >= pd.to_datetime(start_date)) &
                            (df_sp["date"] <= pd.to_datetime(end_date))]
            fig_mkt.add_trace(
                go.Scatter(x=df_sp_f["date"], y=df_sp_f["close"], name="S&P 500"),
                secondary_y=False
            )
        if not df_t.empty:
            df_t_f = df_t[(df_t["date"] >= pd.to_datetime(start_date)) &
                          (df_t["date"] <= pd.to_datetime(end_date))]
            fig_mkt.add_trace(
                go.Scatter(x=df_t_f["date"], y=df_t_f["close"], name=ticker),
                secondary_y=True
            )
        fig_mkt.update_xaxes(title_text="Fecha")
        fig_mkt.update_yaxes(title_text="S&P 500", secondary_y=False)
        fig_mkt.update_yaxes(title_text=f"{ticker} cierre", secondary_y=True)

        # ================
        # FASE 2 - Correlación
        # ================
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
            fig_corr.update_layout(margin=dict(l=60, r=20, t=30, b=40))
        else:
            fig_corr = go.Figure()
            fig_corr.add_annotation(text="Sin matriz de correlación", x=0.5, y=0.5, showarrow=False)

        # ================
        # FASE 2 - ROC
        # ================
        r = load_roc()
        fig_roc = go.Figure()
        if not r.empty:
            if "ticker" in r.columns:
                df_r = r[r["ticker"] == ticker]
            else:
                df_r = r
            if not df_r.empty:
                fig_roc.add_trace(go.Scatter(x=df_r["fpr"], y=df_r["tpr"], name="ROC"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="azar", line=dict(dash="dash")))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate"
                )
            else:
                fig_roc.add_annotation(text="Sin datos ROC para este ticker", x=0.5, y=0.5, showarrow=False)
        else:
            fig_roc.add_annotation(text="Sin datos de ROC", x=0.5, y=0.5, showarrow=False)

        # ================
        # FASE 2 - Métricas clasificación
        # ================
        cls = load_classification_metrics()
        if not cls.empty:
            if "ticker" in cls.columns:
                cls_t = cls[cls["ticker"] == ticker]
            else:
                cls_t = cls

            if not cls_t.empty:
                cols_show = [c for c in cls_t.columns if c not in ["support"]]  # ajusta si quieres
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

        # ================
        # FASE 2 - Recomendación final
        # ================
        rec_text = "Sin recomendación disponible."
        if not signals_all_cache.empty:
            df_sig = signals_all_cache[signals_all_cache["ticker"] == ticker].copy()

            df_sig["date"] = pd.to_datetime(df_sig["date"])

            if pd.api.types.is_datetime64tz_dtype(df_sig["date"]):
                df_sig["date"] = df_sig["date"].dt.tz_convert(None)

            df_sig = df_sig[(df_sig["date"] >= pd.to_datetime(start_date)) &
                            (df_sig["date"] <= pd.to_datetime(end_date))]
            if not df_sig.empty:
                last = df_sig.sort_values("date").iloc[-1]
                sig = last.get("signal", 0)
                proba = last.get("y_proba", None)

                if sig == 1:
                    rec = "COMPRAR / MANTENER"
                elif sig == 0:
                    rec = "NO POSICIÓN"
                else:
                    rec = "VENDER / CERRAR POSICIÓN"

                rec_text = f"Para {last['date'].date()} el modelo recomienda: {rec}"
                if proba is not None:
                    rec_text += f" (probabilidad de subida: {float(proba):.2%})"

        rec_card = html.Div([
            html.P(rec_text),
            html.Ul([
                html.Li("La recomendación se basa en el modelo entrenado sobre los datos históricos."),
                html.Li("Úsalo como apoyo, no como sustituto de tu criterio de inversión.")
            ], style={"fontSize": "11px", "color": "#555"})
        ])

        return (
            fig1, fig2, fig3, fig4,
            fig_bt, metrics_cards, fig_mkt, fig_corr, fig_roc,
            table, rec_card
        )

    except Exception as e:
        # Fallback seguro: figuras vacías + mensajes de error en children
        print("ERROR en update_all:\n", traceback.format_exc())
        empty_fig = go.Figure()
        err_msg = html.P(f"Error en el callback: {e}")

        return (
            empty_fig, empty_fig, empty_fig, empty_fig,  # 4 gráficos Fase 1
            empty_fig, err_msg,                         # backtest fig + métricas children
            empty_fig, empty_fig, empty_fig,            # mercado, corr, roc
            err_msg, err_msg                            # cls-metrics children, recommendation children
        )

if __name__ == "__main__":
    print("Iniciando servidor Dash...")
    try:
        app.run(debug=True)
    except Exception as e:
        print("Error al iniciar la app:", e)
        traceback.print_exc()