#!/usr/bin/env python3
"""Run the notebook's random-forest simulation on the latest session.

Stocks use the latest regular session, 9:30 AM–4:00 PM America/New_York.
Crypto symbols that end in -USD use the last 24 hours. The account is simulated.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from broker import BrokerError, format_connection, open_broker

STOP_PCT = 0.005
TARGET_PCT = 0.01
STARTING_CASH = 10_000.0
SHARES_PER_TRADE = 10
FUTURE_LOOK_AHEAD = 5
READINESS_THRESHOLD = 3
EMA_FAST = 12
EMA_SLOW = 26
RSI_LEN = 14
ADX_LEN = 14
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

FEATURE_COLUMNS = [
    "Close", "Open", "High", "Low", "Volume",
    "1m_fast_ema", "1m_slow_ema", "1m_rsi", "1m_adx", "1m_plus_di", "1m_minus_di", "1m_vwap",
    "15m_fast_ema_htf", "15m_slow_ema_htf", "15m_rsi_htf", "15m_adx_htf",
    "15m_plus_di_htf", "15m_minus_di_htf", "15m_vwap_htf",
    "crossUp", "crossDown", "longArmed", "shortArmed", "vwapLongOK", "vwapShortOK",
    "sweepLow", "sweepHigh", "trapLongOK", "trapShortOK", "bullLadder", "bearLadder",
    "equalHighPool", "equalLowPool", "bosUp", "bosDown", "chochUp", "chochDown",
    "bullFVG", "bearFVG", "inBullFVG", "inBearFVG", "inOB",
]


def is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith("-USD")


def is_future(ticker: str) -> bool:
    return ticker.upper().endswith("=F")


def tune(ticker: str) -> dict:
    params = {"adxMin": 20, "diSpreadMin": 5, "entryWindowBars": 3, "pullbackTicks": 10}
    if ticker.upper() == "AAPL":
        params.update(adxMin=25, diSpreadMin=8, entryWindowBars=5, pullbackTicks=15)
    return params


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Fetching 1m and 15m data for {ticker} over 7d...")
    one_minute = yf.download(ticker, period="7d", interval="1m", progress=False, auto_adjust=True)
    fifteen = yf.download(ticker, period="7d", interval="15m", progress=False, auto_adjust=True)
    if one_minute.empty or fifteen.empty:
        raise SystemExit(f"No price data returned for {ticker}.")
    one_minute = flatten_columns(one_minute)
    fifteen = flatten_columns(fifteen)
    one_minute.index = one_minute.index.tz_convert("America/New_York")
    fifteen.index = fifteen.index.tz_convert("America/New_York")
    return one_minute, fifteen


def add_ema_rsi(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df[f"{prefix}fast_ema"] = ta.trend.ema_indicator(df["Close"], window=EMA_FAST)
    df[f"{prefix}slow_ema"] = ta.trend.ema_indicator(df["Close"], window=EMA_SLOW)
    df[f"{prefix}rsi"] = ta.momentum.rsi(df["Close"], window=RSI_LEN)
    return df


def add_adx_vwap(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    enough = df["Close"].dropna().shape[0] >= ADX_LEN * 2 + 1
    if enough:
        df[f"{prefix}adx"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=ADX_LEN)
        df[f"{prefix}plus_di"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"], window=ADX_LEN)
        df[f"{prefix}minus_di"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"], window=ADX_LEN)
    else:
        df[f"{prefix}adx"] = np.nan
        df[f"{prefix}plus_di"] = np.nan
        df[f"{prefix}minus_di"] = np.nan
    if df["Volume"].dropna().shape[0] >= ADX_LEN:
        df[f"{prefix}vwap"] = ta.volume.volume_weighted_average_price(
            high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=ADX_LEN
        )
    else:
        df[f"{prefix}vwap"] = np.nan
    df["hlc3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    return df


def merge_higher_timeframe(main: pd.DataFrame, higher: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "15m_fast_ema", "15m_slow_ema", "15m_rsi",
        "15m_adx", "15m_plus_di", "15m_minus_di", "15m_vwap",
    ]
    resampled = higher[columns].resample("1min").ffill()
    resampled = resampled.rename(columns={col: f"{col}_htf" for col in columns})
    merged = main.merge(resampled, how="left", left_index=True, right_index=True)
    merged.ffill(inplace=True)
    return merged


def select_window(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, str]:
    if is_crypto(ticker):
        end = df.index[-1]
        start = end - pd.DateOffset(hours=24)
        window = df[df.index >= start].copy()
        label = f"crypto 24h ending {end}"
    elif is_future(ticker):
        last = df.index[-1]
        session_day = last.date() if last.hour < 18 else (last + pd.Timedelta(days=1)).date()
        start = pd.Timestamp(session_day, tz=last.tz) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
        stop = pd.Timestamp(session_day, tz=last.tz) + pd.Timedelta(hours=17)
        window = df[(df.index >= start) & (df.index < stop)].copy()
        label = f"futures session {start:%Y-%m-%d %H:%M} to {stop:%Y-%m-%d %H:%M} America/New_York"
    else:
        rth = df.between_time("09:30", "16:00")
        if rth.empty:
            raise SystemExit(f"No regular-session bars for {ticker}.")
        session_day = rth.index.max().date()
        window = rth[rth.index.date == session_day].copy()
        label = f"equity session {session_day} 09:30-16:00 America/New_York"
    if window.empty:
        raise SystemExit(f"No bars in the selected window for {ticker}.")
    window["is_sym_ok"] = True
    window["is_sess_ok"] = True
    window["is_trade_allowed"] = True
    return window, label


def add_structure(df: pd.DataFrame, entry_window: int) -> pd.DataFrame:
    tick = 0.01
    df["vwapLongOK"] = df["Close"] > df["1m_vwap"]
    df["vwapShortOK"] = df["Close"] < df["1m_vwap"]
    prior_low = df["Low"].rolling(window=entry_window).min().shift(1)
    prior_high = df["High"].rolling(window=entry_window).max().shift(1)
    df["sweepLow"] = (df["Low"] < prior_low) & (df["Close"] > prior_low)
    df["sweepHigh"] = (df["High"] > prior_high) & (df["Close"] < prior_high)
    df["trapLongOK"] = df["sweepLow"] & (df["Close"] > df["1m_fast_ema"])
    df["trapShortOK"] = df["sweepHigh"] & (df["Close"] < df["1m_fast_ema"])

    bull = (df["High"] > df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
    bear = (df["High"] < df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))
    df["bullLadder"] = bull.rolling(window=2).sum() == 2
    df["bearLadder"] = bear.rolling(window=2).sum() == 2
    df["equalHighPool"] = (df["High"] - df["High"].shift(1)).abs() <= tick * 2
    df["equalLowPool"] = (df["Low"] - df["Low"].shift(1)).abs() <= tick * 2
    prev_max = df["High"].rolling(window=5).max().shift(1)
    prev_min = df["Low"].rolling(window=5).min().shift(1)
    df["bosUp"] = (df["Close"] > prev_max) & (df["Close"].shift(1) <= prev_max.shift(1))
    df["bosDown"] = (df["Close"] < prev_min) & (df["Close"].shift(1) >= prev_min.shift(1))
    df["chochUp"] = df["sweepLow"] & (df["Close"] > df["Open"])
    df["chochDown"] = df["sweepHigh"] & (df["Close"] < df["Open"])
    df["bullFVG"] = df["Low"].shift(2) > df["High"].shift(1)
    df["bearFVG"] = df["High"].shift(2) < df["Low"].shift(1)
    df["inBullFVG"] = df["bullFVG"] & (df["Close"] > df["High"].shift(1)) & (df["Close"] < df["Low"].shift(2))
    df["inBearFVG"] = df["bearFVG"] & (df["Close"] < df["Low"].shift(1)) & (df["Close"] > df["High"].shift(2))
    df["inOB"] = (df["Close"] >= df["Low"].shift(5)) & (df["Close"] <= df["High"].shift(5))
    return df


def add_scores(df: pd.DataFrame, tuned: dict) -> pd.DataFrame:
    adx_min = tuned["adxMin"]
    di_spread = tuned["diSpreadMin"]
    entry_window = tuned["entryWindowBars"]
    fast_prev = df["1m_fast_ema"].shift(1)
    slow_prev = df["1m_slow_ema"].shift(1)
    df["crossUp"] = (df["1m_fast_ema"] > df["1m_slow_ema"]) & (fast_prev <= slow_prev)
    df["crossDown"] = (df["1m_fast_ema"] < df["1m_slow_ema"]) & (fast_prev >= slow_prev)
    df["longArmed"] = df["crossUp"] | (
        (df["hlc3"] > df["1m_fast_ema"]) & df["crossUp"].rolling(window=entry_window).max().fillna(False)
    )
    df["shortArmed"] = df["crossDown"] | (
        (df["hlc3"] < df["1m_fast_ema"]) & df["crossDown"].rolling(window=entry_window).max().fillna(False)
    )
    long_score = pd.DataFrame({
        "ma": df["crossUp"],
        "rsi": df["1m_rsi"] < RSI_OVERSOLD,
        "vwap": df["vwapLongOK"],
        "htf": df["15m_fast_ema_htf"] > df["15m_slow_ema_htf"],
        "adx": (df["1m_adx"] > adx_min) & (df["1m_plus_di"] > df["1m_minus_di"]) & ((df["1m_plus_di"] - df["1m_minus_di"]) > di_spread),
        "structure": df["trapLongOK"] | df["chochUp"] | df["bosUp"] | df["bullLadder"],
        "fvg": df["inBullFVG"] | df["inOB"],
    })
    short_score = pd.DataFrame({
        "ma": df["crossDown"],
        "rsi": df["1m_rsi"] > RSI_OVERBOUGHT,
        "vwap": df["vwapShortOK"],
        "htf": df["15m_fast_ema_htf"] < df["15m_slow_ema_htf"],
        "adx": (df["1m_adx"] > adx_min) & (df["1m_minus_di"] > df["1m_plus_di"]) & ((df["1m_minus_di"] - df["1m_plus_di"]) > di_spread),
        "structure": df["trapShortOK"] | df["chochDown"] | df["bosDown"] | df["bearLadder"],
        "fvg": df["inBearFVG"] | df["inOB"],
    })
    df["confScore_long"] = long_score.sum(axis=1)
    df["confScore_short"] = short_score.sum(axis=1)
    df["readyScore"] = df[["confScore_long", "confScore_short"]].max(axis=1)
    df["gateByReadiness"] = (df["readyScore"] >= READINESS_THRESHOLD) & df["is_trade_allowed"]
    df["baseLong"] = (df["confScore_long"] >= READINESS_THRESHOLD) & df["is_trade_allowed"]
    df["baseShort"] = (df["confScore_short"] >= READINESS_THRESHOLD) & df["is_trade_allowed"]
    df["final_signal"] = "HOLD"
    df.loc[df["baseLong"], "final_signal"] = "BUY"
    df.loc[df["baseShort"], "final_signal"] = "SELL"
    df.loc[~df["gateByReadiness"], "final_signal"] = "HOLD"
    return df


def train_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    features = [col for col in FEATURE_COLUMNS if col in df.columns]
    frame = df.copy()
    frame["future_close"] = frame["Close"].shift(-FUTURE_LOOK_AHEAD)
    frame["price_goes_up_future"] = frame["future_close"] > frame["Close"]
    model_rows = frame[features + ["price_goes_up_future"]].dropna()
    if len(model_rows) < 5:
        raise SystemExit(f"Only {len(model_rows)} usable bars. The forest needs at least 5.")
    x = model_rows[features]
    y = model_rows["price_goes_up_future"].astype(bool)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    accuracy = float(model.score(x_test, y_test))
    predictions = pd.Series(model.predict(x), index=x.index).map({True: "BUY", False: "SELL"})
    traded = frame.loc[x.index].copy()
    traded["ml_signal"] = predictions
    return traded, accuracy


def position_quantity(price: float, cash: float) -> float:
    if SHARES_PER_TRADE * price <= cash:
        return float(SHARES_PER_TRADE)
    return cash / price


def simulate(df: pd.DataFrame) -> dict:
    cash = STARTING_CASH
    shares = 0.0
    position_open = False
    position_type = None
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    required = ["Close", "High", "Low", "ml_signal", "is_trade_allowed"]
    rows = df.dropna(subset=[col for col in required if col in df.columns])
    if rows.empty:
        return {
            "cash": cash,
            "profit": 0.0,
            "trades": [],
            "ml_counts": {},
            "last_price": 0.0,
        }

    for index, row in rows.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        signal = row["ml_signal"]
        allowed = bool(row["is_trade_allowed"])
        exited = False
        if position_open and allowed:
            if position_type == "LONG" and low <= stop_loss:
                cash += shares * stop_loss
                trades.append({"action": "STOP_LOSS_LONG", "price": stop_loss, "quantity": shares, "time": str(index)})
                shares = 0.0
                position_open = False
                exited = True
            elif position_type == "LONG" and high >= take_profit:
                cash += shares * take_profit
                trades.append({"action": "TAKE_PROFIT_LONG", "price": take_profit, "quantity": shares, "time": str(index)})
                shares = 0.0
                position_open = False
                exited = True
            elif position_type == "SHORT" and high >= stop_loss:
                cash -= abs(shares) * stop_loss
                trades.append({"action": "STOP_LOSS_SHORT", "price": stop_loss, "quantity": abs(shares), "time": str(index)})
                shares = 0.0
                position_open = False
                exited = True
            elif position_type == "SHORT" and low <= take_profit:
                cash -= abs(shares) * take_profit
                trades.append({"action": "TAKE_PROFIT_SHORT", "price": take_profit, "quantity": abs(shares), "time": str(index)})
                shares = 0.0
                position_open = False
                exited = True
        if exited or position_open or not allowed:
            continue
        quantity = position_quantity(close, STARTING_CASH)
        if signal == "BUY" and cash >= quantity * close:
            cash -= quantity * close
            shares = quantity
            position_open = True
            position_type = "LONG"
            stop_loss = close * (1 - STOP_PCT)
            take_profit = close * (1 + TARGET_PCT)
            trades.append({"action": "BUY_LONG", "price": close, "quantity": quantity, "time": str(index)})
        elif signal == "SELL":
            cash += quantity * close
            shares = -quantity
            position_open = True
            position_type = "SHORT"
            stop_loss = close * (1 + STOP_PCT)
            take_profit = close * (1 - TARGET_PCT)
            trades.append({"action": "OPEN_SHORT", "price": close, "quantity": quantity, "time": str(index)})

    last_price = float(rows["Close"].iloc[-1])
    if position_open and position_type == "LONG":
        cash += shares * last_price
        trades.append({"action": "FORCED_SELL_LONG", "price": last_price, "quantity": shares, "time": str(rows.index[-1])})
        shares = 0.0
    elif position_open and position_type == "SHORT":
        cash -= abs(shares) * last_price
        trades.append({"action": "FORCED_COVER_SHORT", "price": last_price, "quantity": abs(shares), "time": str(rows.index[-1])})
        shares = 0.0
    return {
        "cash": cash,
        "profit": cash - STARTING_CASH,
        "trades": trades,
        "ml_counts": rows["ml_signal"].value_counts().to_dict(),
        "last_price": last_price,
        "shares": shares,
    }


def run(ticker: str, broker_name: str) -> None:
    try:
        connection = open_broker(broker_name).connect()
    except BrokerError as exc:
        raise SystemExit(str(exc)) from exc
    print(format_connection(connection))
    print("Orders stay in the simulation. This run does not send them to the broker.")
    tuned = tune(ticker)
    one_minute, fifteen = download(ticker)
    one_minute = add_ema_rsi(one_minute, "1m_")
    fifteen = add_ema_rsi(fifteen, "15m_")
    one_minute = add_adx_vwap(one_minute, "1m_")
    fifteen = add_adx_vwap(fifteen, "15m_")
    one_minute = merge_higher_timeframe(one_minute, fifteen)
    window, label = select_window(one_minute, ticker)
    window = add_structure(window, tuned["entryWindowBars"])
    window = add_scores(window, tuned)
    traded, accuracy = train_signals(window)
    result = simulate(traded)
    print(f"Session: {label}")
    print(f"Bars: {len(traded)}")
    print(f"Open: {float(traded['Close'].iloc[0]):.2f}  Last: {result['last_price']:.2f}")
    print(f"Forest test accuracy: {accuracy:.3f}")
    print(f"ML signals: {result['ml_counts']}")
    print(f"Starting cash: ${STARTING_CASH:.2f}")
    print(f"Final cash: ${result['cash']:.2f}")
    print(f"Profit/loss: ${result['profit']:.2f}")
    print(f"Trades: {len(result['trades'])}")
    for trade in result["trades"]:
        print(f"  {trade['time']}  {trade['action']}  qty={trade['quantity']:.6g}  price={trade['price']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the notebook random-forest bot.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Yahoo symbol, such as AAPL or BTC-USD")
    parser.add_argument("--broker", choices=["paper", "tradelocker"], default="paper")
    args = parser.parse_args()
    run(args.ticker.upper(), args.broker)


if __name__ == "__main__":
    sys.exit(main())
