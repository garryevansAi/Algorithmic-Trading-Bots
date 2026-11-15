#!/usr/bin/env python3
"""Train one forest on stocks, ETFs, and futures, plus an option-chain forest.

Stocks and ETFs use the latest regular session. Futures symbols ending in =F use
the latest 18:00-to-17:00 America/New_York session. The option model uses the
nearest expiration. Neither run sends a broker order.
"""

from __future__ import annotations

import sys

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from run_bot import (
    FEATURE_COLUMNS,
    FUTURE_LOOK_AHEAD,
    add_adx_vwap,
    add_ema_rsi,
    add_scores,
    add_structure,
    download,
    merge_higher_timeframe,
    select_window,
    simulate,
    tune,
)

STOCKS = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN"]
ETFS = ["SPY", "QQQ", "IWM", "DIA"]
FUTURES = ["ES=F", "NQ=F", "YM=F", "GC=F", "CL=F"]
OPTION_UNDERLIERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "META"]


def build_session(ticker: str) -> tuple[pd.DataFrame, str]:
    tuned = tune(ticker)
    one_minute, fifteen = download(ticker)
    one_minute = add_ema_rsi(one_minute, "1m_")
    fifteen = add_ema_rsi(fifteen, "15m_")
    one_minute = add_adx_vwap(one_minute, "1m_")
    fifteen = add_adx_vwap(fifteen, "15m_")
    merged = merge_higher_timeframe(one_minute, fifteen)
    window, label = select_window(merged, ticker)
    window = add_structure(window, tuned["entryWindowBars"])
    window = add_scores(window, tuned)
    window["symbol"] = ticker
    return window, label


def labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    features = [col for col in FEATURE_COLUMNS if col in frame.columns]
    rows = frame.copy()
    rows["future_close"] = rows["Close"].shift(-FUTURE_LOOK_AHEAD)
    rows["price_goes_up_future"] = rows["future_close"] > rows["Close"]
    keep = list(dict.fromkeys(features + ["price_goes_up_future", "symbol", "High", "Low", "Close", "is_trade_allowed"]))
    return rows[keep].dropna()


def train_price_model(frames: list[pd.DataFrame]) -> None:
    rows = pd.concat([labeled_rows(frame) for frame in frames], ignore_index=True)
    features = [col for col in FEATURE_COLUMNS if col in rows.columns]
    x = rows[features]
    y = rows["price_goes_up_future"].astype(bool)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    test = x_test.copy()
    test["symbol"] = rows.loc[x_test.index, "symbol"]
    test["actual"] = y_test.to_numpy()
    test["predicted"] = model.predict(x_test)
    print(f"\nPooled bars: {len(rows)}  train: {len(x_train)}  test: {len(x_test)}")
    print(f"Pooled test accuracy: {model.score(x_test, y_test):.3f}")
    print(f"{'Symbol':<8} {'Kind':<8} {'Bars':>6} {'Test acc':>9} {'Sim cash':>10} {'P/L':>9}")
    kind = {name: "stock" for name in STOCKS} | {name: "etf" for name in ETFS} | {name: "future" for name in FUTURES}
    for frame in frames:
        symbol = frame["symbol"].iloc[0]
        held = test[test["symbol"] == symbol]
        accuracy = float((held["predicted"] == held["actual"]).mean()) if len(held) else float("nan")
        symbol_rows = rows[rows["symbol"] == symbol].copy()
        symbol_rows["ml_signal"] = pd.Series(model.predict(symbol_rows[features]), index=symbol_rows.index).map(
            {True: "BUY", False: "SELL"}
        )
        symbol_rows["is_trade_allowed"] = True
        result = simulate(symbol_rows)
        print(
            f"{symbol:<8} {kind[symbol]:<8} {len(symbol_rows):>6} {accuracy:>9.3f} "
            f"{result['cash']:>10.2f} {result['profit']:>9.2f}"
        )
    if any(frame["symbol"].iloc[0] in FUTURES for frame in frames):
        print("Futures cash uses the quote in a $10,000 account, not the exchange contract multiplier.")


def option_chain(symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    expirations = list(ticker.options)
    if not expirations:
        raise RuntimeError("no expirations")
    expiration = expirations[0]
    spot_history = ticker.history(period="1d")
    if spot_history.empty:
        raise RuntimeError("no spot")
    spot = float(spot_history["Close"].iloc[-1])
    chain = ticker.option_chain(expiration)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["is_call"] = 1
    puts["is_call"] = 0
    contracts = pd.concat([calls, puts], ignore_index=True)
    contracts["moneyness"] = contracts["strike"] / spot
    contracts["spread"] = contracts["ask"] - contracts["bid"]
    contracts["dte"] = (pd.Timestamp(expiration) - pd.Timestamp.now(tz="America/New_York").tz_localize(None)).days
    contracts["symbol"] = symbol
    contracts["spot"] = spot
    contracts["expiration"] = expiration
    near = contracts[contracts["moneyness"].between(0.85, 1.15)].copy()
    near = near[(near["impliedVolatility"] > 0) & near["volume"].fillna(0).ge(0)]
    median_iv = float(near["impliedVolatility"].median())
    near["high_iv"] = near["impliedVolatility"] > median_iv
    near["call_volume"] = float(calls["volume"].fillna(0).sum())
    near["put_volume"] = float(puts["volume"].fillna(0).sum())
    return near


def train_option_model(chains: list[pd.DataFrame]) -> None:
    rows = pd.concat(chains, ignore_index=True)
    features = ["moneyness", "dte", "volume", "openInterest", "spread", "is_call"]
    rows = rows.dropna(subset=features + ["high_iv"])
    x = rows[features].fillna(0)
    y = rows["high_iv"].astype(bool)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    print(f"\nOption contracts: {len(rows)}  train: {len(x_train)}  test: {len(x_test)}")
    print(f"Option-chain test accuracy: {model.score(x_test, y_test):.3f}")
    print("Target: implied volatility above that chain's median. This is not an option profit.")
    ranked = sorted(zip(features, model.feature_importances_), key=lambda item: item[1], reverse=True)
    print("Feature importance: " + ", ".join(f"{name}={score:.3f}" for name, score in ranked))
    print(f"{'Symbol':<8} {'Expiry':<12} {'Spot':>10} {'Contracts':>10} {'Put/call vol':>13}")
    for chain in chains:
        symbol = chain["symbol"].iloc[0]
        put_volume = float(chain["put_volume"].iloc[0])
        call_volume = float(chain["call_volume"].iloc[0])
        ratio = put_volume / call_volume if call_volume else float("nan")
        print(
            f"{symbol:<8} {chain['expiration'].iloc[0]:<12} {float(chain['spot'].iloc[0]):>10.2f} "
            f"{len(chain):>10} {ratio:>13.2f}"
        )


def main() -> int:
    frames = []
    print("Stocks:", ", ".join(STOCKS))
    print("ETFs:", ", ".join(ETFS))
    print("Futures:", ", ".join(FUTURES))
    for ticker in STOCKS + ETFS + FUTURES:
        try:
            frame, label = build_session(ticker)
        except Exception as exc:
            print(f"Skipped {ticker}: {exc}")
            continue
        print(f"Loaded {ticker}: {label}, {len(frame)} bars")
        frames.append(frame)
    if len(frames) < 2:
        print("Not enough symbols to train a shared model.", file=sys.stderr)
        return 1
    train_price_model(frames)

    chains = []
    print("\nOption underliers:", ", ".join(OPTION_UNDERLIERS))
    for ticker in OPTION_UNDERLIERS:
        try:
            chain = option_chain(ticker)
        except Exception as exc:
            print(f"Skipped {ticker} chain: {exc}")
            continue
        if chain.empty:
            print(f"Skipped {ticker} chain: no contracts near the spot")
            continue
        print(f"Loaded {ticker} chain: {len(chain)} contracts")
        chains.append(chain)
    if len(chains) < 2:
        print("Not enough option chains to train.", file=sys.stderr)
        return 1
    train_option_model(chains)
    return 0


if __name__ == "__main__":
    sys.exit(main())
