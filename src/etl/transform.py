import pandas as pd
import numpy as np

# Install 'ta' library if not already installed
#pip install ta

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import os
import yaml
from datetime import datetime, timezone # Import timezone here

# Configuración
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'UNH', 'XOM']
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
INPUT_DIR = 'data/raw_parquet'
OUTPUT_DIR = 'data/processed'
OUTPUT_DIR_ML = 'data/processed/ml_ready'
METADATA_DIR = 'data/metadata'


def load_raw_data(ticker):
    """Carga datos OHLCV raw de un ticker."""
    file_path = os.path.join(INPUT_DIR, f'{ticker}_ohlcv.parquet')
    df = pd.read_parquet(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def calculate_technical_indicators(df):
    """Calcula indicadores técnicos sobre histórico completo."""
    # Tendencia
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()

    # Momentum
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    # Volatilidad
    bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    df['ATR_14'] = AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).average_true_range()

    # Volumen
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

    return df


def calculate_additional_features(df):
    """Calcula features adicionales: returns, volatilidad, volume_change."""
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Volume_change'] = df['Volume'].pct_change()
    return df


def create_target(df):
    """Crea variable Y: 1 si Close(t+1) > Close(t), 0 en caso contrario."""
    df['Close_next'] = df['Close'].shift(-1)
    df['Target'] = (df['Close_next'] > df['Close']).astype(int)
    df = df.drop(columns=['Close_next'])
    return df


def filter_dates(df, start_date, end_date):
    """Filtra fechas a 2015-2024."""
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    return df[mask].reset_index(drop=True)


def clean_features(df):
    """Limpia NaN pero PRESERVA las columnas de precio OHLCV."""
    # ELIMINAR SOLO columnas que no necesitamos
    # Ya NO eliminamos OHLCV - los preservamos para la aplicación

    # Eliminar solo columnas innecesarias
    extra_cols = ['Dividends', 'Stock Splits']
    df = df.drop(columns=[col for col in extra_cols if col in df.columns], errors='ignore')

    # Eliminar NaN
    df = df.dropna().reset_index(drop=True)
    return df


def save_features_with_prices(df, ticker):
    """Guarda features INCLUYENDO columnas de precio OHLCV."""
    output_file = os.path.join(OUTPUT_DIR, f'{ticker}_features.parquet')

    # Definir el orden deseado de columnas
    base_columns = ['Ticker', 'Date']

    # Columnas de precio OHLCV que debemos preservar
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Columnas de indicadores técnicos
    indicator_columns = [
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
        'MACD', 'MACD_signal', 'MACD_diff',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
        'ATR_14', 'OBV', 'Returns', 'Volatility_10', 'Volume_change', 'Target'
    ]

    # Combinar todas las columnas (solo las que existen en el DataFrame)
    all_columns = base_columns + price_columns + indicator_columns
    existing_columns = [col for col in all_columns if col in df.columns]

    # Reordenar el DataFrame
    df_sorted = df[existing_columns]

    # Guardar
    df_sorted.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')

    # Información de depuración
    price_cols_found = [col for col in price_columns if col in df.columns]
    print(f"  Precios preservados: {price_cols_found}")

    return df_sorted


def transform_ticker(ticker):
    """Pipeline completo para un ticker - PRESERVANDO PRECIOS."""
    print(f"Procesando {ticker}")

    # 1. Cargar histórico completo
    df = load_raw_data(ticker)
    total_original = len(df)

    # 2. Calcular indicadores sobre histórico completo
    df = calculate_technical_indicators(df)
    df = calculate_additional_features(df)
    df = create_target(df)

    print(f"{ticker}: Indicadores calculados sobre {total_original} días")

    # 3. Filtrar fechas 2015-2024
    df = filter_dates(df, START_DATE, END_DATE)

    # 4. Limpiar (SOLO eliminar columnas innecesarias y NaN, PRESERVAR precios)
    df = clean_features(df)

    # 5. Guardar INCLUYENDO columnas de precio
    df_final = save_features_with_prices(df, ticker)

    # Balance de Y
    y_counts = df_final['Target'].value_counts()
    pct_buy = (y_counts.get(1, 0) / len(df_final) * 100) if len(df_final) > 0 else 0
    pct_sell = (y_counts.get(0, 0) / len(df_final) * 100) if len(df_final) > 0 else 0

    print(f"{ticker}: {len(df_final)} días (2015-2024), Y balance: Comprar {pct_buy:.1f}% / Vender {pct_sell:.1f}%")

    return df_final


def create_metadata(stats, combined_df):
    """Crea metadata de la transformación."""
    metadata_file = os.path.join(METADATA_DIR, 'transformation_log.yaml')

    total_buy = int(combined_df['Target'].sum())
    total_sell = int((combined_df['Target'] == 0).sum())

    # Contar columnas de precio preservadas
    price_columns_preserved = [col for col in combined_df.columns if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    metadata = {
        'transformation_date': datetime.now(timezone.utc).isoformat() + 'Z',
        'date_range': {'start': START_DATE, 'end': END_DATE},
        'tickers': TICKERS,
        'total_rows': len(combined_df),
        'price_columns_preserved': price_columns_preserved,
        'technical_indicators': [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
            'MACD', 'MACD_signal', 'MACD_diff',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'OBV', 'ATR_14'
        ],
        'additional_features': ['Returns', 'Volatility_10', 'Volume_change'],
        'total_features': len(combined_df.columns) - 3,
        'target_definition': 'Y = 1 si Close(t+1) > Close(t), 0 en caso contrario',
        'target_distribution': {
            'Comprar (1)': total_buy,
            'Vender (0)': total_sell,
            'Balance': f"{total_buy/len(combined_df)*100:.1f}% / {total_sell/len(combined_df)*100:.1f}%"
        },
        'ticker_stats': stats
    }

    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def main():
    """Ejecuta pipeline para todos los tickers."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_ML, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("\nTransformación y Feature Engineering")
    print(f"Periodo final: {START_DATE} a {END_DATE}")
    print(f"Tickers: {len(TICKERS)}")
    print("✅ PRESERVANDO columnas de precio OHLCV para la aplicación\n")

    all_data = []
    stats = {}

    for ticker in TICKERS:
        try:
            df = transform_ticker(ticker)
            all_data.append(df)

            stats[ticker] = {
                'rows': len(df),
                'features': len(df.columns) - 3,
                'price_columns': [col for col in df.columns if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']],
                'target_distribution': {
                    'Comprar': int(df['Target'].sum()),
                    'Vender': int((df['Target'] == 0).sum())
                }
            }
        except Exception as e:
            print(f"Error en {ticker}: {e}")

    # Consolidar
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(OUTPUT_DIR_ML, 'features_combined.parquet')
        combined.to_parquet(combined_file, index=False, engine='pyarrow', compression='snappy')

        create_metadata(stats, combined)

        # Resumen
        total_buy = combined['Target'].sum()
        total_sell = (combined['Target'] == 0).sum()

        price_cols_count = len([col for col in combined.columns if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

        print(f"\nRESUMEN")
        print(f"Total filas: {len(combined):,}")
        print(f"Total columnas: {len(combined.columns)}")
        print(f"Columnas de precio preservadas: {price_cols_count}")
        print(f"Y balance global: Comprar {total_buy/len(combined)*100:.1f}% / Vender {total_sell/len(combined)*100:.1f}%")
        print(f"Archivos: {OUTPUT_DIR}/")
    else:
        print("❌ No se procesaron datos correctamente")


if __name__ == "__main__":
    main()