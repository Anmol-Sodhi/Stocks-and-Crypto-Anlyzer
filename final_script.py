import getpass  # to securely prompt for password in a terminal
import os
import sys
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import logging
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import matplotlib
matplotlib.use('Agg')  # Set backend here
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import argparse










# Configure Logging
logging.basicConfig(
    filename='stock_analysis.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Email Configuration
SENDER_EMAIL = ''       # Replace with your sender email
SENDER_PASSWORD = ''       # Replace with your email app password
RECIPIENT_EMAILS = []










# Fetch S&P 500 Tickers
def fetch_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia.

    Returns:
        list: List of ticker symbols.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Adjust tickers with dots (e.g., BRK.B) and ensure uppercase without whitespace
        tickers = [ticker.replace('.', '-').strip().upper() for ticker in tickers]
        logging.info('Fetched S&P 500 tickers successfully.')
        return tickers
    except Exception as e:
        logging.error(f'Error fetching S&P 500 tickers: {e}')
        return []

def fetch_crypto_tickers():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 20,
            "page": 1,
            "sparkline": "false"
        }
        response = requests.get(url, params=params)
        data = response.json()
        # Convert retrieved crypto symbols into Yahoo Finance compatible tickers
        # For instance, BTC -> BTC-USD
        # Use a predefined list to ensure correctness
        crypto_tickers = [
            "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "SOL-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
            "SHIB-USD", "TRX-USD", "AVAX-USD", "UNI-USD", "LINK-USD", "ATOM-USD", "XLM-USD", "XMR-USD", "ALGO-USD", "BCH-USD",
            "FTM-USD", "APE-USD", "AAVE-USD", "MKR-USD", "RPL-USD", "ARB-USD", "OP-USD", "QNT-USD", "LDO-USD", "FIL-USD",
            "NEAR-USD", "VET-USD", "EGLD-USD", "ICP-USD", "SAND-USD", "MANA-USD", "GRT-USD", "CAKE-USD", "EOS-USD", "CHZ-USD",
            "LUNC-USD", "THETA-USD", "TUSD-USD", "USDP-USD", "WBTC-USD", "BUSD-USD", "GT-USD", "OKB-USD", "CRV-USD", "PAXG-USD",
            "GMX-USD", "XDC-USD", "KLAY-USD", "APT-USD", "SNX-USD", "FTT-USD", "DASH-USD", "CAW-USD", "1INCH-USD", "ENJ-USD",
            "RUNE-USD", "ZIL-USD", "ZEC-USD", "XEC-USD", "CELO-USD", "STX-USD", "MINA-USD", "GALA-USD", "IMX-USD", "HNT-USD",
            "FLOW-USD", "WAVES-USD", "NEO-USD", "CRO-USD", "XCH-USD", "AMP-USD", "WAXP-USD", "CEEK-USD", "ANKR-USD", "RAY-USD",
            "DAG-USD", "IOTX-USD", "KAVA-USD", "BSV-USD", "CKB-USD", "RSR-USD", "OMG-USD", "TFUEL-USD", "RVN-USD", "HT-USD",
            "NANO-USD", "WOO-USD", "SUSHI-USD", "BAT-USD", "BAL-USD", "GNO-USD", "COMP-USD"
        ]
        # Ensure all crypto tickers are uppercase and stripped
        crypto_tickers = [ticker.strip().upper() for ticker in crypto_tickers]
        logging.info('Fetched Crypto tickers successfully.')
        return crypto_tickers
    except Exception as e:
        logging.error(f'Error fetching crypto tickers: {e}')
        return []
    


def detect_loss_of_momentum(df, past_x_days=30, rsi_threshold=50, macd_signal_distance=0.005):
    """
    Detects a 'Loss of Momentum' pattern based on RSI and MACD indicators within the past_x_days.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data with 'Close' prices.
        past_x_days (int): Number of past days to consider for detection.
        rsi_threshold (float): RSI value below which momentum is considered lost.
        macd_signal_distance (float): Threshold for MACD and Signal line crossover.
    
    Returns:
        bool: True if 'Loss of Momentum' is detected within past_x_days, False otherwise.
    """
    try:
        # Ensure DataFrame is sorted by date
        df = df.sort_index()

        # Slice the DataFrame to include only the last past_x_days
        df = df.tail(past_x_days)

        if len(df) < 2:
            # Not enough data to detect momentum loss
            return False

        # Calculate RSI
        df['RSI'] = calculate_RSI(df['Close'])
        df = df.dropna(subset=['RSI'])

        # Calculate MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Iterate through the sliced DataFrame to check for any 'Loss of Momentum'
        for i in range(1, len(df)):
            latest_rsi = df['RSI'].iloc[i]
            latest_macd = df['MACD'].iloc[i]
            latest_signal = df['Signal_Line'].iloc[i]

            previous_rsi = df['RSI'].iloc[i-1]
            previous_macd = df['MACD'].iloc[i-1]
            previous_signal = df['Signal_Line'].iloc[i-1]

            # Conditions for Loss of Momentum
            rsi_condition = latest_rsi < rsi_threshold
            macd_cross_down = previous_macd > previous_signal and latest_macd < latest_signal

            # Debugging Outputs
            print(f"Day {i}: RSI={latest_rsi:.2f}, MACD={latest_macd:.5f}, Signal Line={latest_signal:.5f}")
            print(f"Day {i}: RSI Condition Met: {rsi_condition}, MACD Cross Down: {macd_cross_down}")

            if rsi_condition and macd_cross_down:
                return True  # Loss of Momentum detected within past_x_days

        return False  # No Loss of Momentum detected within past_x_days

    except Exception as e:
        logging.error(f"Error detecting Loss of Momentum: {e}")
        print(f"Error detecting Loss of Momentum: {e}", flush=True)
        return False



    






# Email Sending Function
def send_email(subject, body, to_emails):
    """
    Sends an email using SMTP with App Password authentication.

    Args:
        subject (str): Subject of the email.
        body (str): Body content of the email (can include HTML).
        to_emails (list): List of recipient email addresses.
    """
    try:
        # Set up the SMTP server
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        # Create the email
        message = MIMEMultipart()
        message['From'] = SENDER_EMAIL
        message['To'] = ', '.join(to_emails)
        message['Subject'] = subject

        # Attach the body with HTML formatting
        html_body = f"<html><body>{body}</body></html>"
        message.attach(MIMEText(html_body, 'html'))

        # Send the email
        server.sendmail(SENDER_EMAIL, to_emails, message.as_string())
        server.quit()

        print(f"Email sent successfully to {', '.join(to_emails)}.")
        logging.info(f"Email sent successfully to {', '.join(to_emails)}.")
    except Exception as e:
        print(f"Failed to send email: {e}")
        logging.error(f"Failed to send email: {e}")

# Define Analysis Functions

# Function to calculate Fibonacci retracement levels

def calculate_fibonacci(data, peak, trough):
    """
    Calculates Fibonacci retracement levels between a peak and trough.

    Args:
        data (pd.DataFrame): The stock data containing 'High' and 'Low' columns.
        peak (int): The index of the peak.
        trough (int): The index of the trough.

    Returns:
        dict: Dictionary containing Fibonacci levels.
    """
    try:
        # Use position-based indexing with .iloc to avoid KeyError
        high_price = data['High'].iloc[peak]
        low_price = data['Low'].iloc[trough]
        
        # Debugging: Print types and values to ensure they are scalars
        print(f"calculate_fibonacci - High Price: {high_price} (Type: {type(high_price)})")
        print(f"calculate_fibonacci - Low Price: {low_price} (Type: {type(low_price)})")
        
        # Ensure high_price and low_price are scalar floats
        # Ensure high_price and low_price are scalar floats
        high_price = get_scalar(high_price)
        low_price = get_scalar(low_price)

        # Calculate the difference between the high and low prices
        diff = high_price - low_price

        # Define Fibonacci levels
        levels = {
            '0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '100%': low_price
        }
        return levels
    except Exception as e:
        logging.error(f"Error in calculate_fibonacci for data indices peak={peak}, trough={trough}: {e}")
        print(f"Error in calculate_fibonacci for data indices peak={peak}, trough={trough}: {e}")
        return {}




# Candlestick Pattern Detection Function (Weekly)
def detect_weekly_candlestick_pattern(df):
    """
    Detects specific candlestick patterns in the stock data on a weekly timeframe.

    Returns:
        dict: Contains 'Bullish' and 'Bearish' keys with True/False values.
    """
    # Ensure DataFrame is sorted by date
    df = df.sort_index()

    # Handle missing data through interpolation
    df = df.interpolate(method='time')

    # Identify Red and Green Candles
    df['IsRed'] = df['Close'] < df['Open']
    df['IsGreen'] = df['Close'] > df['Open']

    # Calculate Candle Ranges
    df['Range'] = df['High'] - df['Low']

    # Use the last four candles
    if len(df) < 4:
        return {'Bullish': False, 'Bearish': False}

    window = df.iloc[-4:]

    # Set epsilon for floating-point comparisons
    epsilon = 1e-5

    # Extract ranges as scalar floats using get_scalar
    range_1 = get_scalar(window.iloc[0]['Range'])
    range_2 = get_scalar(window.iloc[1]['Range'])
    range_3 = get_scalar(window.iloc[2]['Range'])
    range_4 = get_scalar(window.iloc[3]['Range'])

    #### Bullish Pattern Detection ####
    is_red_1 = get_scalar(window.iloc[0]['IsRed'])
    is_red_2 = get_scalar(window.iloc[1]['IsRed'])
    is_red_3 = get_scalar(window.iloc[2]['IsRed'])
    is_green_4 = get_scalar(window.iloc[3]['IsGreen'])

    # Ranges strictly increasing
    ranges_increasing_bullish = (
        (range_1 + epsilon) < range_2 and
        (range_2 + epsilon) < range_3
    )

    # Engulfing condition for bullish pattern
    high_3 = get_scalar(window.iloc[2]['High'])
    low_3 = get_scalar(window.iloc[2]['Low'])
    high_4 = get_scalar(window.iloc[3]['High'])
    low_4 = get_scalar(window.iloc[3]['Low'])
    is_engulfed_bullish = (
        (high_4 <= high_3 + epsilon) and
        (low_4 >= low_3 - epsilon)
    )

    # Fourth candle's range less than third candle's range
    range_condition_bullish = (range_4 <= range_3 + epsilon)

    # Check bullish pattern
    bullish_pattern = (
        is_red_1 and is_red_2 and is_red_3 and
        ranges_increasing_bullish and is_green_4 and
        range_condition_bullish and is_engulfed_bullish
    )

    #### Bearish Pattern Detection ####
    is_green_1 = get_scalar(window.iloc[0]['IsGreen'])
    is_green_2 = get_scalar(window.iloc[1]['IsGreen'])
    is_green_3 = get_scalar(window.iloc[2]['IsGreen'])
    is_red_4 = get_scalar(window.iloc[3]['IsRed'])

    # Ranges strictly increasing
    ranges_increasing_bearish = (
        (range_1 + epsilon) < range_2 and
        (range_2 + epsilon) < range_3
    )

    # Engulfing condition for bearish pattern
    is_engulfed_bearish = (
        (high_4 <= high_3 + epsilon) and
        (low_4 >= low_3 - epsilon)
    )

    # Fourth candle's range less than third candle's range
    range_condition_bearish = (range_4 <= range_3 + epsilon)

    # Check bearish pattern
    bearish_pattern = (
        is_green_1 and is_green_2 and is_green_3 and
        ranges_increasing_bearish and is_red_4 and
        range_condition_bearish and is_engulfed_bearish
    )

    # Clean up temporary columns
    df.drop(columns=['IsRed', 'IsGreen', 'Range'], inplace=True)

    return {'Bullish': bullish_pattern, 'Bearish': bearish_pattern}





def detect_monthly_candlestick_pattern(df):
    df = df.sort_index()
    df = df.interpolate(method='time')

    df['IsRed'] = df['Close'] < df['Open']
    df['IsGreen'] = df['Close'] > df['Open']
    df['Body'] = abs(df['Close'] - df['Open'])

    if len(df) < 3:
        return {'Bullish': False, 'Bearish': False}

    window = df.iloc[-3:]  # last 3 monthly candles
    epsilon = 1e-5

    # Convert everything to plain floats/bools
    body_1 = get_scalar(window.iloc[0]['Body'])  # float
    body_2 = get_scalar(window.iloc[1]['Body'])
    body_3 = get_scalar(window.iloc[2]['Body'])

    # If any is None, bail out
    if body_1 is None or body_2 is None or body_3 is None:
        return {'Bullish': False, 'Bearish': False}

    is_green_1 = bool(get_scalar(window.iloc[0]['IsGreen']))
    is_green_2 = bool(get_scalar(window.iloc[1]['IsGreen']))
    is_green_3 = bool(get_scalar(window.iloc[2]['IsGreen']))

    is_red_1 = bool(get_scalar(window.iloc[0]['IsRed']))
    is_red_2 = bool(get_scalar(window.iloc[1]['IsRed']))
    is_red_3 = bool(get_scalar(window.iloc[2]['IsRed']))

    # Now these comparisons are safe
    # Bodies strictly decreasing for bullish
    bodies_decreasing_bullish = (
        (body_1 > body_2 + epsilon) and
        (body_2 > body_3 + epsilon)
    )
    bullish_pattern = (is_green_1 and is_green_2 and is_green_3 and bodies_decreasing_bullish)

    # Bodies strictly decreasing for bearish
    bodies_decreasing_bearish = (
        (body_1 > body_2 + epsilon) and
        (body_2 > body_3 + epsilon)
    )
    bearish_pattern = (is_red_1 and is_red_2 and is_red_3 and bodies_decreasing_bearish)

    # Clean up
    df.drop(columns=['IsRed', 'IsGreen', 'Body'], inplace=True, errors='ignore')
    return {'Bullish': bullish_pattern, 'Bearish': bearish_pattern}





# Helper function to calculate RSI (if not already included)
def calculate_RSI(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi





def get_scalar(value):
    """
    Extracts a scalar value from a pandas Series, DataFrame, or scalar input.

    Args:
        value: A pandas Series, DataFrame, or scalar value.

    Returns:
        The scalar value extracted from the Series/DataFrame or the original scalar.
        Returns None if the input cannot be reduced to a single scalar.
    """
    try:
        # If value is a Series with a single element, return that element
        if isinstance(value, pd.Series):
            if len(value) == 1:
                return value.iloc[0]
            else:
                logging.warning(f"Expected a single value, but got a Series with multiple values: {value}")
                return None
        
        # If value is a DataFrame with one cell, return that cell
        elif isinstance(value, pd.DataFrame):
            if value.size == 1:
                return value.iloc[0, 0]
            else:
                logging.warning(f"Expected a single value, but got a DataFrame with multiple cells: {value}")
                return None

        # If the value is already scalar (int, float, str, etc.), return it directly
        else:
            return value
    except Exception as e:
        logging.error(f"Error extracting scalar value: {e}")
        return None






def analyze_ticker(ticker, past_x_days=30, max_retries=3, retry_delay=2):
    """
    Analyzes a single ticker across multiple timeframes and indicators.
    Adds checks for a 12-month low within past_x_days and a 1-month high within past_x_days.
    Implements retry logic for data retrieval.

    Args:
        ticker (str): The ticker symbol to analyze.
        past_x_days (int): Number of past days to consider for certain analyses.
        max_retries (int): Maximum number of retries for data download.
        retry_delay (int): Delay in seconds between retries.

    Returns:
        dict: Analysis results containing signals for Long and Short positions,
              plus references to daily data and fib levels, if computed.
    """
    try:
        print(f"Analyzing {ticker}...", flush=True)
        logging.info(f"Analyzing {ticker}...")
        result = {
            'Ticker': ticker,
            'Long': False,
            'Short': False,
            'Signals': [],
            'data_daily': None,               # Will store daily dataframe if available
            'fib_levels_daily': None,         # Will store daily fib levels if computed
            'fib_levels_monthly': None        # Will store monthly fib levels if computed
        }

        def download_with_retries(ticker, period, interval, retries=max_retries, delay=retry_delay):
            for attempt in range(retries):
                try:
                    print(f"Attempt {attempt + 1}: Downloading {interval} data for {ticker}...", flush=True)
                    data = yf.download(ticker, period=period, interval=interval,
                                       progress=False, auto_adjust=True)
                    if not data.empty:
                        print(f"Successfully downloaded {interval} data for {ticker} on attempt {attempt + 1}.", flush=True)
                        return data
                    else:
                        print(f"Attempt {attempt + 1}: No data retrieved for {ticker} with interval {interval}. Retrying...", flush=True)
                except Exception as e:
                    print(f"Attempt {attempt + 1}: Error downloading {ticker} with interval {interval}: {e}. Retrying...", flush=True)
                time.sleep(delay)
            logging.warning(f"Failed to download {interval} data for {ticker} after {retries} attempts.")
            return pd.DataFrame()

        # ─────────────────────────────────────────────────────────────────
        # Fetch DAILY data
        # ─────────────────────────────────────────────────────────────────
        data_daily = download_with_retries(ticker, period='1y', interval='1d')
        if data_daily.empty:
            logging.warning(f"No daily data for {ticker}.")
            print(f"No daily data for {ticker}.", flush=True)
        else:
            print(f"Downloaded daily data for {ticker}: {data_daily.shape[0]} rows.", flush=True)
            print(data_daily.head(), flush=True)
            while isinstance(data_daily.columns, pd.MultiIndex) and data_daily.columns.nlevels > 1:
                data_daily.columns = data_daily.columns.droplevel(0)
            print("Flattened daily columns:", data_daily.columns)

            # If only 5 columns, rename them
            if len(data_daily.columns) == 5:
                data_daily.columns = ["Open", "High", "Low", "Close", "Volume"]
                print("Renamed daily columns:", data_daily.columns)

        # ─────────────────────────────────────────────────────────────────
        # Fetch WEEKLY data
        # ─────────────────────────────────────────────────────────────────
        data_weekly = download_with_retries(ticker, period='1y', interval='1wk')
        if data_weekly.empty:
            logging.warning(f"No weekly data for {ticker}.")
            print(f"No weekly data for {ticker}.", flush=True)
        else:
            print(f"Downloaded weekly data for {ticker}: {data_weekly.shape[0]} rows.", flush=True)
            print(data_weekly.head(), flush=True)
            while isinstance(data_weekly.columns, pd.MultiIndex) and data_weekly.columns.nlevels > 1:
                data_weekly.columns = data_weekly.columns.droplevel(0)
            print("Flattened weekly columns:", data_weekly.columns)

            if len(data_weekly.columns) == 5:
                data_weekly.columns = ["Open", "High", "Low", "Close", "Volume"]
                print("Renamed weekly columns:", data_weekly.columns)

        # ─────────────────────────────────────────────────────────────────
        # Fetch MONTHLY data
        # ─────────────────────────────────────────────────────────────────
        data_monthly = yf.download(ticker, period='5y', interval='1mo', progress=False, auto_adjust=True)
        if data_monthly.empty:
            logging.warning(f"No monthly data for {ticker}.")
            print(f"No monthly data for {ticker}.", flush=True)
        else:
            print(f"Downloaded monthly data for {ticker}: {data_monthly.shape[0]} rows.", flush=True)
            print(data_monthly.head(), flush=True)
            while isinstance(data_monthly.columns, pd.MultiIndex) and data_monthly.columns.nlevels > 1:
                data_monthly.columns = data_monthly.columns.droplevel(0)
            print("Flattened monthly columns:", data_monthly.columns)

            if len(data_monthly.columns) == 5:
                data_monthly.columns = ["Open", "High", "Low", "Close", "Volume"]
                print("Renamed monthly columns:", data_monthly.columns)

        # ─────────────────────────────────────────────────────────────────
        # 12-MONTH & 1-MONTH LOGIC
        # ─────────────────────────────────────────────────────────────────
        lookback_days_12m = 251
        lookback_days_1m = 21

        # Check 12-month low
        if not data_daily.empty and len(data_daily) >= lookback_days_12m:
            recent_period_12m = data_daily.iloc[-past_x_days:]
            lowest_12m = data_daily['Low'].rolling(window=lookback_days_12m, min_periods=1).min()
            recent_12m_low = lowest_12m.iloc[-past_x_days:].min()
            current_low = recent_period_12m['Low'].min()

            current_low = get_scalar(current_low)
            recent_12m_low = get_scalar(recent_12m_low)

            print(f"{ticker}: Recent 12-month low within past {past_x_days} days: {current_low} <= {recent_12m_low}", flush=True)
            if current_low <= recent_12m_low + 1e-8:
                signal = f"12-month low within the past {past_x_days} days"
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)
        else:
            print(f"{ticker}: Not enough daily data to evaluate 12-month low.", flush=True)

        # Check 1-month high
        if not data_daily.empty and len(data_daily) >= lookback_days_1m:
            recent_period_1m = data_daily.iloc[-past_x_days:]
            highest_1m = data_daily['High'].rolling(window=lookback_days_1m, min_periods=1).max()
            recent_1m_high = highest_1m.iloc[-past_x_days:].max()
            current_high = recent_period_1m['High'].max()

            current_high = get_scalar(current_high)
            recent_1m_high = get_scalar(recent_1m_high)

            print(f"{ticker}: Recent 1-month high within past {past_x_days} days: {current_high} >= {recent_1m_high}", flush=True)
            if current_high >= recent_1m_high - 1e-8:
                signal = f"1-month high within the past {past_x_days} days"
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)
        else:
            print(f"{ticker}: Not enough daily data to evaluate 1-month high.", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # DAILY EMA & RSI Analysis
        # ─────────────────────────────────────────────────────────────────
        if not data_daily.empty:
            # EMAs
            data_daily['EMA_9'] = data_daily['Close'].ewm(span=9, adjust=False).mean()
            data_daily['EMA_200'] = data_daily['Close'].ewm(span=200, adjust=False).mean()

            # RSI
            data_daily['RSI'] = calculate_RSI(data_daily['Close'])

            # Candle coloring
            data_daily['IsRed'] = data_daily['Close'] < data_daily['Open']
            data_daily['IsGreen'] = data_daily['Close'] > data_daily['Open']

            # Consecutive candles
            data_daily['ConsecutiveRed'] = data_daily['IsRed'].groupby(
                (data_daily['IsRed'] != data_daily['IsRed'].shift()).cumsum()
            ).cumcount() + 1
            data_daily['ConsecutiveGreen'] = data_daily['IsGreen'].groupby(
                (data_daily['IsGreen'] != data_daily['IsGreen'].shift()).cumsum()
            ).cumcount() + 1

            last_row = data_daily.iloc[-1]
            last_close = get_scalar(last_row['Close'])
            ema_9 = get_scalar(last_row['EMA_9'])
            ema_200 = get_scalar(last_row['EMA_200'])
            rsi = get_scalar(last_row['RSI'])
            consecutive_red = get_scalar(last_row['ConsecutiveRed'])
            consecutive_green = get_scalar(last_row['ConsecutiveGreen'])

            is_red_val = get_scalar(last_row['IsRed'])
            is_red = bool(is_red_val) if isinstance(is_red_val, bool) else False
            is_green_val = get_scalar(last_row['IsGreen'])
            is_green = bool(is_green_val) if isinstance(is_green_val, bool) else False

            print(f"{ticker}: Last Daily Close: {last_close}, EMA_9: {ema_9}, EMA_200: {ema_200}, RSI: {rsi}", flush=True)
            print(f"{ticker}: Consecutive Red Candles: {consecutive_red}, Consecutive Green Candles: {consecutive_green}", flush=True)

            ema_threshold = 0.01
            rsi_overbought = 65
            consecutive_green_candles = 5

            # Near EMA 9
            if abs(last_close - ema_9) / last_close < ema_threshold:
                if is_red and consecutive_red >= 7:
                    signal = f"Price near EMA 9 on Daily chart after 7 red days within past {past_x_days} days (Possible Reversal Zone)"
                    result['Long'] = True
                    result['Signals'].append(signal)
                    print(f"Signal for {ticker}: {signal}", flush=True)
                elif is_green and consecutive_green >= 7:
                    signal = f"Price near EMA 9 on Daily chart after 7 green days within past {past_x_days} days (Possible Reversal Zone)"
                    result['Short'] = True
                    result['Signals'].append(signal)
                    print(f"Signal for {ticker}: {signal}", flush=True)

            # Near 200-day EMA & Overbought
            if abs(last_close - ema_200) / last_close < ema_threshold:
                overbought_rsi = rsi > rsi_overbought
                overbought_candles = (consecutive_green >= consecutive_green_candles)
                print(f"{ticker}: Overbought RSI: {overbought_rsi}, Overbought Candles: {overbought_candles}", flush=True)
                if overbought_rsi or overbought_candles:
                    signal = f"Price has reached the 200-day EMA after moving into overbought conditions within past {past_x_days} days"
                    if overbought_rsi:
                        signal += f" (RSI > {rsi_overbought})"
                    if overbought_candles:
                        signal += f" ({consecutive_green} consecutive green candles)"
                    result['Short'] = True
                    result['Signals'].append(signal)
                    print(f"Signal for {ticker}: {signal}", flush=True)

            # Oops Pattern (New 52-week low + bullish reversal)
            if len(data_daily) >= 252:
                rolling_low = data_daily['Low'].rolling(window=252, min_periods=1).min()
                current_low_value = get_scalar(last_row['Low'])
                rolling_low_value = get_scalar(rolling_low.iloc[-1])
                if last_close == current_low_value and current_low_value == rolling_low_value:
                    print(f"{ticker}: New 52-week low detected.", flush=True)
                    is_reversal = detect_bullish_reversal_candle(data_daily)
                    print(f"{ticker}: Bullish reversal candle detected: {is_reversal}", flush=True)
                    if is_reversal:
                        signal = f"Oops Pattern Detected: New 52-week low with bullish reversal candle within past {past_x_days} days"
                        result['Long'] = True
                        result['Signals'].append(signal)
                        print(f"Signal for {ticker}: {signal}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # Climax Pattern Detection (Stocks & Cryptos)
        # ─────────────────────────────────────────────────────────────────
        if not data_daily.empty:
            print(f"{ticker}: Checking for Climax Pattern within past {past_x_days} days...", flush=True)
            climax_pattern = detect_climax_pattern(data_daily)
            if climax_pattern == 'Bearish':
                signal = f"Climax Top detected within past {past_x_days} days (Bearish Reversal)"
                result['Short'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)
            elif climax_pattern == 'Bullish':
                signal = f"Climax Bottom detected within past {past_x_days} days (Bullish Reversal)"
                result['Long'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # LOSS OF MOMENTUM PATTERN DETECTION
        # ─────────────────────────────────────────────────────────────────
        if not data_daily.empty:
            print(f"{ticker}: Checking for Loss of Momentum pattern within past {past_x_days} days...", flush=True)
            loss_momentum = detect_loss_of_momentum(data_daily, past_x_days=past_x_days)
            if loss_momentum:
                signal = f"Loss of Momentum detected based on RSI and MACD on Daily chart within past {past_x_days} days"
                result['Short'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # WEEKLY EMA Analysis
        # ─────────────────────────────────────────────────────────────────
        if not data_weekly.empty:
            data_weekly['EMA_9'] = data_weekly['Close'].ewm(span=9, adjust=False).mean()
            data_weekly['IsRed'] = data_weekly['Close'] < data_weekly['Open']
            data_weekly['IsGreen'] = data_weekly['Close'] > data_weekly['Open']

            data_weekly['ConsecutiveRed'] = data_weekly['IsRed'].groupby(
                (data_weekly['IsRed'] != data_weekly['IsRed'].shift()).cumsum()
            ).cumcount() + 1
            data_weekly['ConsecutiveGreen'] = data_weekly['IsGreen'].groupby(
                (data_weekly['IsGreen'] != data_weekly['IsGreen'].shift()).cumsum()
            ).cumcount() + 1

            threshold = 0.01
            last_row_weekly = data_weekly.iloc[-1]
            last_close_weekly = get_scalar(last_row_weekly['Close'])
            ema_9_weekly = get_scalar(last_row_weekly['EMA_9'])

            consecutive_red_weekly = get_scalar(last_row_weekly['ConsecutiveRed'])
            consecutive_green_weekly = get_scalar(last_row_weekly['ConsecutiveGreen'])

            is_red_weekly_val = get_scalar(last_row_weekly['IsRed'])
            is_green_weekly_val = get_scalar(last_row_weekly['IsGreen'])

            is_red_weekly = bool(is_red_weekly_val) if isinstance(is_red_weekly_val, bool) else False
            is_green_weekly = bool(is_green_weekly_val) if isinstance(is_green_weekly_val, bool) else False

            print(f"{ticker}: Last Weekly Close: {last_close_weekly}, EMA_9: {ema_9_weekly}", flush=True)
            print(f"{ticker}: Consecutive Red Weeks: {consecutive_red_weekly}, Consecutive Green Weeks: {consecutive_green_weekly}", flush=True)

            if last_close_weekly is not None and pd.notna(last_close_weekly) and last_close_weekly != 0:
                ratio = abs(last_close_weekly - ema_9_weekly) / last_close_weekly
                if ratio < threshold:
                    if is_red_weekly and consecutive_red_weekly >= 3:
                        signal = f"Price near EMA 9 on Weekly chart after 3 red weeks within past {past_x_days} days (Possible Reversal Zone)"
                        result['Long'] = True
                        result['Signals'].append(signal)
                        print(f"Signal for {ticker}: {signal}", flush=True)
                    elif is_green_weekly and consecutive_green_weekly >= 3:
                        signal = f"Price near EMA 9 on Weekly chart after 3 green weeks within past {past_x_days} days (Possible Reversal Zone)"
                        result['Short'] = True
                        result['Signals'].append(signal)
                        print(f"Signal for {ticker}: {signal}", flush=True)
            else:
                print(f"{ticker}: Skipping Weekly EMA check (last_close_weekly is zero or NaN).", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # MONTHLY EMA Analysis
        # ─────────────────────────────────────────────────────────────────
        if not data_monthly.empty:
            data_monthly['EMA_9'] = data_monthly['Close'].ewm(span=9, adjust=False).mean()
            data_monthly['IsRed'] = data_monthly['Close'] < data_monthly['Open']
            data_monthly['IsGreen'] = data_monthly['Close'] > data_monthly['Open']

            data_monthly['ConsecutiveRed'] = data_monthly['IsRed'].groupby(
                (data_monthly['IsRed'] != data_monthly['IsRed'].shift()).cumsum()
            ).cumcount() + 1
            data_monthly['ConsecutiveGreen'] = data_monthly['IsGreen'].groupby(
                (data_monthly['IsGreen'] != data_monthly['IsGreen'].shift()).cumsum()
            ).cumcount() + 1

            threshold = 0.01
            last_row_monthly = data_monthly.iloc[-1]
            last_close_monthly = get_scalar(last_row_monthly['Close'])
            ema_9_monthly = get_scalar(last_row_monthly['EMA_9'])

            consecutive_red_monthly = get_scalar(last_row_monthly['ConsecutiveRed'])
            consecutive_green_monthly = get_scalar(last_row_monthly['ConsecutiveGreen'])

            is_red_monthly_val = get_scalar(last_row_monthly['IsRed'])
            is_green_monthly_val = get_scalar(last_row_monthly['IsGreen'])

            is_red_monthly = bool(is_red_monthly_val) if isinstance(is_red_monthly_val, bool) else False
            is_green_monthly = bool(is_green_monthly_val) if isinstance(is_green_monthly_val, bool) else False

            print(f"{ticker}: Last Monthly Close: {last_close_monthly}, EMA_9: {ema_9_monthly}", flush=True)
            print(f"{ticker}: Consecutive Red Months: {consecutive_red_monthly}, Consecutive Green Months: {consecutive_green_monthly}", flush=True)

            if last_close_monthly != 0.0:
                ratio_monthly = abs(last_close_monthly - ema_9_monthly) / last_close_monthly
                if ratio_monthly < threshold:
                    if is_red_monthly and consecutive_red_monthly >= 2:
                        signal = f"Price near EMA 9 on Monthly chart after 2 red months within past {past_x_days} days (Possible Reversal Zone)"
                        result['Long'] = True
                        result['Signals'].append(signal)
                        print(f"Signal for {ticker}: {signal}", flush=True)
                    elif is_green_monthly and consecutive_green_monthly >= 2:
                        signal = f"Price near EMA 9 on Monthly chart after 2 green months within past {past_x_days} days (Possible Reversal Zone)"
                        result['Short'] = True
                        result['Signals'].append(signal)
                        print(f"Signal for {ticker}: {signal}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # FIBONACCI on DAILY
        # ─────────────────────────────────────────────────────────────────
        if not data_daily.empty:
            high_values = data_daily['High'].values
            low_values = -data_daily['Low'].values
            peaks_daily, _ = find_peaks(high_values, distance=30)
            troughs_daily, _ = find_peaks(low_values, distance=30)

            if len(peaks_daily) > 0 and len(troughs_daily) > 0:
                peak = int(peaks_daily[-1])
                trough = int(troughs_daily[-1])
                print(f"{ticker}: Latest Peak at index {peak} (Price: {data_daily['High'].iloc[peak]}), "
                      f"Latest Trough at index {trough} (Price: {data_daily['Low'].iloc[trough]})", flush=True)
                try:
                    if peak > trough:
                        levels = calculate_fibonacci(data_daily, peak, trough)
                        if levels:
                            # 1) Keep only 38.2%, 50%, 61.8% lines:
                            levels = {
                                k: v for k, v in levels.items()
                                if k in ['38.2%', '50%', '61.8%']
                            }
                            # 2) Save them for charting
                            result['fib_levels_daily'] = levels

                            current_price = get_scalar(data_daily['Close'].iloc[-1])
                            for level_name, level_price in levels.items():
                                threshold = 0.01
                                if abs(current_price - level_price) / current_price < threshold:
                                    # It's a match => trigger signal
                                    result['Long'] = True
                                    signal = f"Price near Fibonacci support at {level_name} on Daily chart within past {past_x_days} days"
                                    result['Signals'].append(signal)
                                    print(f"Signal for {ticker}: {signal}", flush=True)
                                    break
                    else:
                        levels = calculate_fibonacci(data_daily, trough, peak)
                        if levels:
                            # 1) Keep only 38.2%, 50%, 61.8% lines:
                            levels = {
                                k: v for k, v in levels.items()
                                if k in ['38.2%', '50%', '61.8%']
                            }
                            # 2) Save them for charting
                            result['fib_levels_daily'] = levels

                            current_price = get_scalar(data_daily['Close'].iloc[-1])
                            for level_name, level_price in levels.items():
                                threshold = 0.01
                                if abs(current_price - level_price) / current_price < threshold:
                                    # It's a match => trigger signal
                                    result['Short'] = True
                                    signal = f"Price near Fibonacci resistance at {level_name} on Daily chart within past {past_x_days} days"
                                    result['Signals'].append(signal)
                                    print(f"Signal for {ticker}: {signal}", flush=True)
                                    break
                except Exception as e:
                    logging.error(f"Error in calculate_fibonacci for {ticker}: {e}")
                    print(f"Error in calculate_fibonacci for {ticker}: {e}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # FIBONACCI on MONTHLY
        # ─────────────────────────────────────────────────────────────────
        if not data_monthly.empty:
            high_vals_m = data_monthly['High'].values
            low_vals_m = -data_monthly['Low'].values
            peaks_monthly, _ = find_peaks(high_vals_m, distance=3)
            troughs_monthly, _ = find_peaks(low_vals_m, distance=3)

            print(f"{ticker}: Found {len(peaks_monthly)} peaks and {len(troughs_monthly)} troughs in monthly data.", flush=True)

            if len(peaks_monthly) > 0 and len(troughs_monthly) > 0:
                peak = int(peaks_monthly[-1])
                trough = int(troughs_monthly[-1])
                print(f"{ticker}: Latest Monthly Peak at index {peak} (Price: {data_monthly['High'].iloc[peak]}), "
                      f"Latest Monthly Trough at index {trough} (Price: {data_monthly['Low'].iloc[trough]})", flush=True)
                try:
                    if peak > trough:
                        levels = calculate_fibonacci(data_monthly, peak, trough)
                        if levels:
                            # 1) Keep only 38.2%, 50%, 61.8%
                            levels = {
                                k: v for k, v in levels.items()
                                if k in ['38.2%', '50%', '61.8%']
                            }
                            result['fib_levels_monthly'] = levels

                            current_price = get_scalar(data_monthly['Close'].iloc[-1])
                            for level_name, level_price in levels.items():
                                threshold = 0.01
                                if abs(current_price - level_price) / current_price < threshold:
                                    result['Long'] = True
                                    signal = f"Price near Fibonacci support at {level_name} on Monthly chart within past {past_x_days} days"
                                    result['Signals'].append(signal)
                                    print(f"Signal for {ticker}: {signal}", flush=True)
                                    break
                    else:
                        levels = calculate_fibonacci(data_monthly, trough, peak)
                        if levels:
                            # 1) Keep only 38.2%, 50%, 61.8%
                            levels = {
                                k: v for k, v in levels.items()
                                if k in ['38.2%', '50%', '61.8%']
                            }
                            result['fib_levels_monthly'] = levels

                            current_price = get_scalar(data_monthly['Close'].iloc[-1])
                            for level_name, level_price in levels.items():
                                threshold = 0.01
                                if abs(current_price - level_price) / current_price < threshold:
                                    result['Short'] = True
                                    signal = f"Price near Fibonacci resistance at {level_name} on Monthly chart within past {past_x_days} days"
                                    result['Signals'].append(signal)
                                    print(f"Signal for {ticker}: {signal}", flush=True)
                                    break
                except Exception as e:
                    logging.error(f"Error in calculate_fibonacci (Monthly) for {ticker}: {e}")
                    print(f"Error in calculate_fibonacci (Monthly) for {ticker}: {e}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # Candlestick Patterns (Weekly)
        # ─────────────────────────────────────────────────────────────────
        if not data_weekly.empty:
            weekly_patterns = detect_weekly_candlestick_pattern(data_weekly)
            print(f"{ticker}: Weekly Bullish Pattern Detected: {weekly_patterns['Bullish']}, "
                  f"Bearish Pattern Detected: {weekly_patterns['Bearish']}", flush=True)
            if weekly_patterns['Bullish']:
                signal = f"Weekly Bullish Candlestick Pattern Detected within past {past_x_days} days"
                result['Long'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)
            if weekly_patterns['Bearish']:
                signal = f"Weekly Bearish Candlestick Pattern Detected within past {past_x_days} days"
                result['Short'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # Candlestick Patterns (Monthly)
        # ─────────────────────────────────────────────────────────────────
        if not data_monthly.empty:
            monthly_patterns = detect_monthly_candlestick_pattern(data_monthly)
            print(f"{ticker}: Monthly Bullish Pattern Detected: {monthly_patterns['Bullish']}, "
                  f"Bearish Pattern Detected: {monthly_patterns['Bearish']}", flush=True)
            if monthly_patterns['Bullish']:
                signal = f"Monthly Bullish 'Big-Smallest' Pattern Detected within past {past_x_days} days"
                result['Long'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)
            if monthly_patterns['Bearish']:
                signal = f"Monthly Bearish 'Big-Smallest' Pattern Detected within past {past_x_days} days"
                result['Short'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # Climax Pattern Detection (Stocks & Cryptos)
        # ─────────────────────────────────────────────────────────────────
        if not data_daily.empty:
            print(f"{ticker}: Checking for Climax Pattern within past {past_x_days} days...", flush=True)
            climax_pattern = detect_climax_pattern(data_daily)
            if climax_pattern == 'Bearish':
                signal = f"Climax Top detected within past {past_x_days} days (Bearish Reversal)"
                result['Short'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)
            elif climax_pattern == 'Bullish':
                signal = f"Climax Bottom detected within past {past_x_days} days (Bullish Reversal)"
                result['Long'] = True
                result['Signals'].append(signal)
                print(f"Signal for {ticker}: {signal}", flush=True)

        # ─────────────────────────────────────────────────────────────────
        # If both Long & Short got triggered, pick one & remove conflicts
        # ─────────────────────────────────────────────────────────────────
        if result['Long'] and result['Short']:
            new_signals = [
                s for s in result['Signals'] 
                if not ("Bearish" in s and "Loss of Momentum" not in s)
            ]
            # Update the 'Short' flag: retain it only if 'Loss of Momentum' is present
            loss_momentum_present = any("Loss of Momentum" in s for s in new_signals)
            result['Short'] = loss_momentum_present
            result['Signals'] = new_signals

        # Stash daily data reference for later charting
        result['data_daily'] = data_daily
        return result

    except Exception as e:
        logging.error(f"Error analyzing ticker {ticker}: {e}")
        print(f"Error analyzing ticker {ticker}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None







def save_ticker_chart(ticker, data_daily, analysis_result, past_x_days=30, save_folder="charts"):
    """
    Saves a chart for the given ticker using *already-sliced* daily data 
    (last `past_x_days` rows), highlighting relevant signals or lines.
    
    Args:
        ticker (str): The stock/crypto ticker symbol.
        data_daily (pd.DataFrame): The already-sliced daily DataFrame (last N rows).
        analysis_result (dict): The result from analyze_ticker, including fib levels if any.
        past_x_days (int): The number of days used for chart slicing (for chart title).
        save_folder (str): The folder path in which to save the generated chart PNG.
    """
    import matplotlib.pyplot as plt

    # 1) Ensure the output folder exists:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plt.figure(figsize=(10, 6))

    # 2) Plot the daily Close price
    plt.plot(data_daily.index, data_daily['Close'], label='Close Price', color='blue')

    # 3) Plot EMAs
    if 'EMA_9' in data_daily.columns:
        plt.plot(data_daily.index, data_daily['EMA_9'], label='EMA 9', color='orange', linestyle='--')
    if 'EMA_200' in data_daily.columns:
        plt.plot(data_daily.index, data_daily['EMA_200'], label='EMA 200', color='green', linestyle='--')

    # 4) Plot only the relevant Fib lines (38.2%, 50%, 61.8%) if they exist
    possible_fibs = analysis_result.get('fib_levels_daily')  # Could be None or dict
    if not possible_fibs:
        fibs = {}
    else:
        fibs = possible_fibs
    for fib_name, fib_price in fibs.items():
        plt.axhline(y=fib_price, color='red', linestyle=':', alpha=0.7,
                    label=f"Fib {fib_name} = {fib_price:.2f}")

    # 5) Title, etc.
    plt.title(f"{ticker} Daily Chart (Last {past_x_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # 6) Save the figure inside `save_folder`
    filename = os.path.join(save_folder, f"{ticker}_chart.png")
    plt.savefig(filename)
    plt.close()

    print(f"Chart saved for {ticker} as '{filename}'")


def process_tickers(past_x_days=30, test_mode=False, test_tickers=10):
    """
    Processes a list of tickers, analyzing each and collecting signals for Long and Short positions.
    Implements rate limiting and optional testing with a subset of tickers.

    Args:
        past_x_days (int): Number of past days to consider for certain analyses.
        test_mode (bool): If True, processes only a subset of tickers for testing purposes.
        test_tickers (int): Number of tickers to process if test_mode is True.

    Returns:
        None
    """
    # 1) Fetch all tickers
    tickers = fetch_sp500_tickers()
    crypto_tickers = fetch_crypto_tickers()
    tickers = tickers + crypto_tickers

    if not tickers:
        print("No tickers fetched. Exiting.")
        logging.error("No tickers fetched. Exiting.")
        return

    # 2) Optionally limit to a small subset (for testing)
    if test_mode:
        tickers = tickers[:test_tickers]
        print(f"Test Mode: Processing first {test_tickers} tickers.")

    long_positions = []
    short_positions = []

    # 3) Create (or verify) the folder where charts are saved
    save_folder = "charts"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 4) Main loop: analyze each ticker
    total_tickers = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        print(f"\nProcessing {ticker} ({idx}/{total_tickers})", flush=True)

        analysis_result = analyze_ticker(ticker, past_x_days=past_x_days)

        # 5) If we have daily data, slice it to the last `past_x_days` rows, then call save_ticker_chart
        if analysis_result and 'data_daily' in analysis_result:
            data_daily = analysis_result['data_daily']
            if data_daily is not None and not data_daily.empty:
                data_daily = data_daily.iloc[-past_x_days:]  # Only last `past_x_days` rows
                save_ticker_chart(ticker, data_daily, analysis_result,
                                  past_x_days=past_x_days,
                                  save_folder=save_folder)

        # 6) Collect signals for Long/Short lists
        if analysis_result:
            if analysis_result['Long']:
                long_positions.append({
                    'Ticker': ticker,
                    'Signals': analysis_result['Signals']
                })
            if analysis_result['Short']:
                short_positions.append({
                    'Ticker': ticker,
                    'Signals': analysis_result['Signals']
                })

        # 7) Rate limiting
        time.sleep(0.5)

    # 8) Email alerts if any positions are found
    if long_positions or short_positions:
        stock_long_positions = [item for item in long_positions if not item['Ticker'].endswith("-USD")]
        crypto_long_positions = [item for item in long_positions if item['Ticker'].endswith("-USD")]

        stock_short_positions = [item for item in short_positions if not item['Ticker'].endswith("-USD")]
        crypto_short_positions = [item for item in short_positions if item['Ticker'].endswith("-USD")]

        email_body = ""

        # ----- S&P 500 and Stock Alerts -----
        if stock_long_positions or stock_short_positions:
            email_body += "<h1>S&P 500 and Stock Alerts</h1>"

            # Long Positions (Stocks)
            if stock_long_positions:
                email_body += "<h2>Long Position Alerts (Stocks)</h2><ul>"
                for item in stock_long_positions:
                    bullet_points = "</li><li>".join(item['Signals'])
                    email_body += (
                        f"<li><strong>{item['Ticker']}</strong><ul>"
                        f"<li>{bullet_points}</li>"
                        "</ul></li>"
                    )
                email_body += "</ul>"

            # Short Positions (Stocks)
            if stock_short_positions:
                email_body += "<h2>Short Position Alerts (Stocks)</h2><ul>"
                for item in stock_short_positions:
                    bullet_points = "</li><li>".join(item['Signals'])
                    email_body += (
                        f"<li><strong>{item['Ticker']}</strong><ul>"
                        f"<li>{bullet_points}</li>"
                        "</ul></li>"
                    )
                email_body += "</ul>"

        # ----- Crypto Alerts -----
        if crypto_long_positions or crypto_short_positions:
            email_body += "<h1>Crypto Alerts</h1>"

            # Long Positions (Crypto)
            if crypto_long_positions:
                email_body += "<h2>Long Position Alerts (Crypto)</h2><ul>"
                for item in crypto_long_positions:
                    bullet_points = "</li><li>".join(item['Signals'])
                    email_body += (
                        f"<li><strong>{item['Ticker']}</strong><ul>"
                        f"<li>{bullet_points}</li>"
                        "</ul></li>"
                    )
                email_body += "</ul>"

            # Short Positions (Crypto)
            if crypto_short_positions:
                email_body += "<h2>Short Position Alerts (Crypto)</h2><ul>"
                for item in crypto_short_positions:
                    bullet_points = "</li><li>".join(item['Signals'])
                    email_body += (
                        f"<li><strong>{item['Ticker']}</strong><ul>"
                        f"<li>{bullet_points}</li>"
                        "</ul></li>"
                    )
                email_body += "</ul>"

        # ----- Loss of Momentum Alerts -----
        if short_positions:
            # Filter only "Loss of Momentum" signals
            loss_momentum_stocks = [
                item for item in short_positions 
                if any("Loss of Momentum" in signal for signal in item['Signals']) 
                and not item['Ticker'].endswith("-USD")
            ]
            loss_momentum_cryptos = [
                item for item in short_positions 
                if any("Loss of Momentum" in signal for signal in item['Signals']) 
                and item['Ticker'].endswith("-USD")
            ]

            if loss_momentum_stocks or loss_momentum_cryptos:
                email_body += "<h1>Loss of Momentum Alerts</h1>"

                # Loss of Momentum (Stocks)
                if loss_momentum_stocks:
                    email_body += "<h2>Loss of Momentum Alerts (Stocks)</h2><ul>"
                    for item in loss_momentum_stocks:
                        bullet_points = "</li><li>".join(item['Signals'])
                        email_body += (
                            f"<li><strong>{item['Ticker']}</strong><ul>"
                            f"<li>{bullet_points}</li>"
                            "</ul></li>"
                        )
                    email_body += "</ul>"

                # Loss of Momentum (Crypto)
                if loss_momentum_cryptos:
                    email_body += "<h2>Loss of Momentum Alerts (Crypto)</h2><ul>"
                    for item in loss_momentum_cryptos:
                        bullet_points = "</li><li>".join(item['Signals'])
                        email_body += (
                            f"<li><strong>{item['Ticker']}</strong><ul>"
                            f"<li>{bullet_points}</li>"
                            "</ul></li>"
                        )
                    email_body += "</ul>"

        # ----- Climax Pattern Alerts -----  # NEW SECTION
        # Combine long and short positions to check for Climax Patterns
        if long_positions or short_positions:
            # Merge long and short positions for comprehensive search
            combined_positions = long_positions + short_positions

            # Filter for Climax Top and Climax Bottom signals
            climax_stocks = [
                item for item in combined_positions
                if (any("Climax Top detected" in s for s in item['Signals']) or
                    any("Climax Bottom detected" in s for s in item['Signals']))
                and not item['Ticker'].endswith("-USD")
            ]
            climax_cryptos = [
                item for item in combined_positions
                if (any("Climax Top detected" in s for s in item['Signals']) or
                    any("Climax Bottom detected" in s for s in item['Signals']))
                and item['Ticker'].endswith("-USD")
            ]

            if climax_stocks or climax_cryptos:
                email_body += "<h1>Climax Pattern Alerts</h1>"

                # Climax Pattern Alerts (Stocks)
                if climax_stocks:
                    email_body += "<h2>Climax Pattern Alerts (Stocks)</h2><ul>"
                    for item in climax_stocks:
                        # Identify the specific climax signals
                        climax_signals = [s for s in item['Signals'] if "Climax" in s]
                        bullet_points = "</li><li>".join(climax_signals)
                        email_body += (
                            f"<li><strong>{item['Ticker']}</strong><ul>"
                            f"<li>{bullet_points}</li>"
                            "</ul></li>"
                        )
                    email_body += "</ul>"

                # Climax Pattern Alerts (Crypto)
                if climax_cryptos:
                    email_body += "<h2>Climax Pattern Alerts (Crypto)</h2><ul>"
                    for item in climax_cryptos:
                        climax_signals = [s for s in item['Signals'] if "Climax" in s]
                        bullet_points = "</li><li>".join(climax_signals)
                        email_body += (
                            f"<li><strong>{item['Ticker']}</strong><ul>"
                            f"<li>{bullet_points}</li>"
                            "</ul></li>"
                        )
                    email_body += "</ul>"

        # Only send email if email_body is not empty
        if email_body:
            send_email(
                subject=f"Daily Stock and Crypto Alerts for {datetime.now().strftime('%Y-%m-%d')}",
                body=email_body,
                to_emails=RECIPIENT_EMAILS
            )
            logging.info("Email alerts sent successfully.")
        else:
            print("No actionable alerts found today.")
            logging.info("No actionable alerts found today.")





# Helper function to detect bullish reversal candlestick patterns
def detect_bullish_reversal_candle(df):
    """
    Detects common bullish reversal candlestick patterns on daily data.

    Returns:
        bool: True if a bullish reversal pattern is detected, False otherwise.
    """
    # Ensure DataFrame is sorted by date
    df = df.sort_index()

    # Use the last two candles
    if len(df) < 2:
        return False

    prev_candle = df.iloc[-2]
    last_candle = df.iloc[-1]

    # Bullish Engulfing Pattern
    bullish_engulfing = (
        prev_candle['Close'] < prev_candle['Open'] and  # Previous candle is bearish
        last_candle['Close'] > last_candle['Open'] and  # Last candle is bullish
        last_candle['Close'] > prev_candle['Open'] and  # Last close higher than previous open
        last_candle['Open'] < prev_candle['Close']      # Last open lower than previous close
    )

    # Hammer Pattern
    candle_range = last_candle['High'] - last_candle['Low']
    body_size = abs(last_candle['Close'] - last_candle['Open'])
    lower_wick = min(last_candle['Close'], last_candle['Open']) - last_candle['Low']
    hammer = (
        body_size / candle_range < 0.3 and             # Small body
        lower_wick / candle_range > 0.4 and            # Long lower wick
        last_candle['Close'] > last_candle['Open']     # Bullish candle
    )

    # Morning Star Pattern (requires three candles)
    if len(df) < 3:
        morning_star = False
    else:
        third_last_candle = df.iloc[-3]
        morning_star = (
            third_last_candle['Close'] < third_last_candle['Open'] and  # Bearish first candle
            abs(prev_candle['Close'] - prev_candle['Open']) / candle_range < 0.3 and  # Small second candle
            last_candle['Close'] > last_candle['Open'] and  # Bullish third candle
            last_candle['Close'] > (third_last_candle['Open'] + third_last_candle['Close']) / 2  # Close above midpoint of first candle
        )

    # Check if any bullish reversal pattern is detected
    return bullish_engulfing or hammer or morning_star





def detect_climax_pattern(df):
    """
    Detects the Climax Pattern in stock data.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data with 'Close', 'Open', 'High', 'Low', and 'Volume' columns.
    
    Returns:
        str: 'Bullish' if Climax Bottom detected,
             'Bearish' if Climax Top detected,
             None otherwise.
    """
    try:
        # Ensure DataFrame is sorted by date
        df = df.sort_index()
        
        # Consider only the last 10 days for pattern detection
        recent_data = df.tail(10)
        
        if len(recent_data) < 5:
            # Not enough data to detect pattern
            return None
        
        # Calculate the average volume over the last 10 days
        average_volume = recent_data['Volume'].mean()
        
        # Identify days with volume at least 2 times the average volume
        volume_spike = recent_data['Volume'] >= (2 * average_volume)
        
        # Identify sharp price movements (e.g., price change >= 5%)
        recent_data = recent_data.copy()
        recent_data['Price_Change'] = recent_data['Close'].pct_change().abs()

        price_spike = recent_data['Price_Change'] >= 0.05  # 5% price movement
        
        # Combine volume spike and price spike
        climax_days = recent_data[volume_spike & price_spike]
        
        # If at least one day meets both conditions, consider it a Climax Pattern
        if not climax_days.empty:
            # Further check for reversal signs after the climax day
            last_climax_day_index = climax_days.index[-1]
            subsequent_data = df.loc[last_climax_day_index:].tail(3)  # Check next 3 days
            
            # Example reversal signs: Bearish engulfing for Climax Top, Bullish engulfing for Climax Bottom
            if len(subsequent_data) >= 2:
                prev = subsequent_data.iloc[-2]
                last = subsequent_data.iloc[-1]
                
                # Climax Top Reversal: Bearish Engulfing
                bearish_engulfing = (
                    prev['Close'] > prev['Open'] and
                    last['Close'] < last['Open'] and
                    last['Close'] < prev['Open'] and
                    last['Open'] > prev['Close']
                )
                
                # Climax Bottom Reversal: Bullish Engulfing
                bullish_engulfing = (
                    prev['Close'] < prev['Open'] and
                    last['Close'] > last['Open'] and
                    last['Close'] > prev['Open'] and
                    last['Open'] < prev['Close']
                )
                
                if bearish_engulfing:
                    return 'Bearish'  # Climax Top detected
                elif bullish_engulfing:
                    return 'Bullish'  # Climax Bottom detected
        
        return None  # No Climax Pattern detected
    
    except Exception as e:
        logging.error(f"Error detecting Climax Pattern: {e}")
        print(f"Error detecting Climax Pattern: {e}", flush=True)
        return None






# Initialize the argument parser
parser = argparse.ArgumentParser(
    description="Analyze stock and crypto tickers for trading signals."
)

# Add the optional argument for past_x_days
parser.add_argument(
    '-d', '--days',
    type=int,
    help='Number of past days to consider for analyses (default: prompts user)'
)

# Parse the arguments
args = parser.parse_args()







if __name__ == "__main__":
    try:
        # 1) Prompt user for the email credentials and recipients
        sender_email_input = input("Enter the sender's email address: ")
        sender_password_input = getpass.getpass("Enter the sender's email password (app password): ")
        
        # This allows multiple emails, comma-separated
        recipients_input = input("Enter one or more recipient emails (comma-separated): ")
        
        # Parse them into a list
        recipient_emails_list = [email.strip() for email in recipients_input.split(",") if email.strip()]
    
        # 2) Override your global or module-level variables with these inputs
        SENDER_EMAIL = sender_email_input
        SENDER_PASSWORD = sender_password_input
        RECIPIENT_EMAILS = recipient_emails_list
    
        # 3) Determine the number of past_x_days using argparse
        if args.days:
            past_x_days = args.days
            print(f"Number of past days set to {past_x_days} via command-line argument.")
        else:
            user_input = input("Enter the number of past_x_days (e.g., 30): ")
            try:
                past_x_days = int(user_input)
            except ValueError:
                print("Invalid input. Defaulting to 30 days.")
                past_x_days = 30
    
        # 4) Decide whether to run test mode or not (this is optional)
        #    For simplicity, we can just set test_mode=False
        test_mode = False
    
        # 5) Finally, call your main process function
        process_tickers(
            past_x_days=past_x_days,
            test_mode=test_mode
        )
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
        logging.info("Script interrupted by user.")
        sys.exit(0)
