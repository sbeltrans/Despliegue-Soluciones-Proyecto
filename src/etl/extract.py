import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import yaml

# Configuración
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'UNH', 'XOM']
PERIOD = 'max'
INTERVAL = '1d'
OUTPUT_DIR_RAW = 'data/raw_parquet'
OUTPUT_DIR_METADATA = 'data/metadata'


def extract_ohlcv_data(ticker):
    """Extrae datos OHLCV de un ticker usando yfinance."""
    print(f"Extrayendo OHLCV: {ticker}")

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=PERIOD, interval=INTERVAL)

        if hist.empty:
            print(f"Error en {ticker}: No se obtuvieron datos")
            return None

        hist['Ticker'] = ticker
        hist.reset_index(inplace=True)

        # Reordenar columnas
        columns = ['Ticker', 'Date'] + [col for col in hist.columns if col not in ['Ticker', 'Date']]
        hist = hist[columns]

        fecha_inicio = hist['Date'].min().strftime('%Y-%m-%d')
        fecha_fin = hist['Date'].max().strftime('%Y-%m-%d')
        print(f"{ticker}: {len(hist)} días ({fecha_inicio} a {fecha_fin})")

        return hist

    except Exception as e:
        print(f"Error en {ticker}: {e}")
        return None


def save_to_parquet(df, filename):
    """Guarda DataFrame en formato Parquet."""
    df.to_parquet(filename, index=False, engine='pyarrow', compression='snappy')


def create_metadata(ticker_stats):
    """Genera metadata de la extracción en YAML."""
    metadata_file = os.path.join(OUTPUT_DIR_METADATA, 'extraction_log.yaml')
    total_rows = sum(stats['rows'] for stats in ticker_stats.values())

    metadata = {
        'extraction_date': datetime.utcnow().isoformat() + 'Z',
        'period': PERIOD,
        'interval': INTERVAL,
        'total_tickers': len(ticker_stats),
        'total_rows_extracted': total_rows,
        'data_source': 'yfinance - OHLCV',
        'output_format': 'parquet',
        'ticker_details': ticker_stats
    }

    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def main():
    """Función principal de extracción."""
    print("\nExtracción de datos OHLCV - S&P 500")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Periodo: {PERIOD} | Intervalo: {INTERVAL}\n")

    ticker_stats = {}

    for ticker in TICKERS:
        ohlcv_data = extract_ohlcv_data(ticker)

        if ohlcv_data is not None:
            ticker_stats[ticker] = {
                'rows': len(ohlcv_data),
                'date_range': {
                    'start': ohlcv_data['Date'].min().strftime('%Y-%m-%d'),
                    'end': ohlcv_data['Date'].max().strftime('%Y-%m-%d')
                },
                'file': f'{ticker}_ohlcv.parquet'
            }

            ohlcv_file = os.path.join(OUTPUT_DIR_RAW, f'{ticker}_ohlcv.parquet')
            save_to_parquet(ohlcv_data, ohlcv_file)

    create_metadata(ticker_stats)

    # Resumen
    total_rows = sum(stats['rows'] for stats in ticker_stats.values())
    print(f"\nRESUMEN")
    print(f"Tickers exitosos: {len(ticker_stats)}/{len(TICKERS)}")
    print(f"Total días: {total_rows:,}")
    print(f"Archivos: {OUTPUT_DIR_RAW}/")


if __name__ == "__main__":
    main()
