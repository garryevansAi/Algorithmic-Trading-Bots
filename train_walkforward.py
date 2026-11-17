#!/usr/bin/env python3
"""Train on earlier sessions and trade only the next one.

Features stop at the signal bar. The order is filled on the next bar's open.
A five-minute embargo keeps training labels out of the test session. Predictions
between 45% and 55% are holds. Each fill pays one tick of slippage.
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from run_bot import (
    FEATURE_COLUMNS,
    FUTURE_LOOK_AHEAD,
    add_adx_vwap,
    add_ema_rsi,
    add_scores,
    add_structure,
    download,
    is_future,
    merge_higher_timeframe,
    tune,
)

STOCKS = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN"]
ETFS = ["SPY", "QQQ", "IWM", "DIA"]
FUTURES = {
    "ES=F": {"tick": 0.25, "multiplier": 5.0, "name": "MES"},
    "NQ=F": {"tick": 0.25, "multiplier": 2.0, "name": "MNQ"},
    "YM=F": {"tick": 1.0, "multiplier": 0.5, "name": "MYM"},
    "GC=F": {"tick": 0.10, "multiplier": 10.0, "name": "MGC"},
    "CL=F": {"tick": 0.01, "multiplier": 100.0, "name": "MCL"},
}
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
STOP_PCT = 0.005
TARGET_PCT = 0.01


def prepare(ticker: str) -> pd.DataFrame:
    tuned = tune(ticker)
    one_minute, fifteen = download(ticker)
    one_minute = add_ema_rsi(one_minute, "1m_")
    fifteen = add_ema_rsi(fifteen, "15m_")
    one_minute = add_adx_vwap(one_minute, "1m_")
    fifteen = add_adx_vwap(fifteen, "15m_")
    merged = merge_higher_timeframe(one_minute, fifteen)
    merged = add_structure(merged, tuned["entryWindowBars"])
    merged["is_trade_allowed"] = True
    merged = add_scores(merged, tuned)
    merged = merged.sort_index()
    if is_future(ticker):
        session = []
        for ts in merged.index:
            if ts.hour >= 18:
                session.append((pd.Timestamp(ts) + pd.DateOffset(days=1)).date())
            elif ts.hour < 17:
                session.append(ts.date())
            else:
                session.append(pd.NaT)
        merged["session"] = session
        merged = merged[merged["session"].notna()]
    else:
        merged = merged.between_time("09:30", "16:00")
        merged["session"] = [ts.date() for ts in merged.index]
    merged["symbol"] = ticker
    return merged


def with_future_close(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("session", sort=False):
        group = group.copy()
        group["future_close"] = group["Close"].shift(-FUTURE_LOOK_AHEAD)
        parts.append(group)
    return pd.concat(parts).sort_index()


def embargoed_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    labeled = with_future_close(frame)
    sessions = [day for day in pd.unique(labeled["session"]) if pd.notna(day)]
    if len(sessions) < 2:
        raise RuntimeError(f"only {len(sessions)} session")
    test_day = sessions[-1]
    test = labeled[labeled["session"] == test_day]
    train = labeled[labeled["session"] < test_day]
    return train, test, test_day


def signals_for(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, float]:
    features = [col for col in FEATURE_COLUMNS if col in train.columns]
    train_rows = train.dropna(subset=features + ["future_close"]).copy()
    train_rows["up"] = train_rows["future_close"] > train_rows["Close"]
    test_rows = test.dropna(subset=features).copy()
    if len(train_rows) < 100 or len(test_rows) < 5:
        raise RuntimeError("not enough rows")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_rows[features], train_rows["up"].astype(bool))
    scored = test_rows.dropna(subset=["future_close"])
    accuracy = float("nan")
    if len(scored) >= 5:
        actual = scored["future_close"] > scored["Close"]
        predicted = model.predict(scored[features])
        accuracy = float((predicted == actual.to_numpy()).mean())
    proba = model.predict_proba(test_rows[features])
    up_index = list(model.classes_).index(True)
    chance = proba[:, up_index]
    label = pd.Series("HOLD", index=test_rows.index)
    label[chance >= LONG_THRESHOLD] = "BUY"
    label[chance <= SHORT_THRESHOLD] = "SELL"
    return label, accuracy


def simulate_next_bar(frame: pd.DataFrame, signal: pd.Series, slip: float, multiplier: float, quantity: float) -> dict:
    ordered = frame.sort_index()
    stamps = list(ordered.index)
    profit = 0.0
    longs = 0
    shorts = 0
    i = 0
    while i < len(stamps) - 1:
        stamp = stamps[i]
        side = signal.get(stamp, "HOLD")
        if side not in {"BUY", "SELL"}:
            i += 1
            continue
        entry_stamp = stamps[i + 1]
        entry_bar = ordered.loc[entry_stamp]
        raw_open = float(entry_bar["Open"])
        if side == "BUY":
            entry = raw_open + slip
            stop = entry * (1 - STOP_PCT)
            target = entry * (1 + TARGET_PCT)
            longs += 1
        else:
            entry = raw_open - slip
            stop = entry * (1 + STOP_PCT)
            target = entry * (1 - TARGET_PCT)
            shorts += 1
        exit_price = None
        exit_at = len(stamps) - 1
        for j in range(i + 1, len(stamps)):
            bar = ordered.loc[stamps[j]]
            high = float(bar["High"])
            low = float(bar["Low"])
            if side == "BUY" and low <= stop:
                exit_price = stop - slip
                exit_at = j
                break
            if side == "BUY" and high >= target:
                exit_price = target - slip
                exit_at = j
                break
            if side == "SELL" and high >= stop:
                exit_price = stop + slip
                exit_at = j
                break
            if side == "SELL" and low <= target:
                exit_price = target + slip
                exit_at = j
                break
        if exit_price is None:
            last = float(ordered.loc[stamps[-1]]["Close"])
            exit_price = last - slip if side == "BUY" else last + slip
        change = exit_price - entry if side == "BUY" else entry - exit_price
        profit += change * multiplier * quantity
        i = exit_at + 1
    return {"profit": profit, "longs": longs, "shorts": shorts}


def run_symbol(ticker: str) -> dict | None:
    frame = prepare(ticker)
    try:
        train, test, test_day = embargoed_split(frame)
    except RuntimeError as exc:
        print(f"Skipped {ticker}: {exc}")
        return None
    try:
        signal, accuracy = signals_for(train, test)
    except RuntimeError as exc:
        print(f"Skipped {ticker}: {exc}")
        return None
    if is_future(ticker):
        spec = FUTURES[ticker]
        result = simulate_next_bar(test, signal, spec["tick"], spec["multiplier"], 1)
        unit = spec["name"]
        kind = "future"
    else:
        result = simulate_next_bar(test, signal, 0.01, 1.0, 10)
        unit = "10 shares"
        kind = "etf" if ticker in ETFS else "stock"
    held = int((signal == "HOLD").sum())
    print(
        f"{ticker:<8} {kind:<7} {test_day}  train={train['session'].nunique():>2}  "
        f"acc={accuracy:.3f}  long={result['longs']:>3}  short={result['shorts']:>3}  "
        f"hold_bars={held:>4}  pnl={result['profit']:>8.2f}  ({unit})"
    )
    return {"ticker": ticker, "kind": kind, "day": test_day, **result, "accuracy": accuracy, "unit": unit}


def main() -> int:
    print("Walk-forward test: earlier sessions train, the last session is traded.")
    print("Entry is the next bar's open. Slippage is one tick per side. Hold band is 45% to 55%.")
    print(f"{'Symbol':<8} {'Kind':<7} test session")
    results = []
    for ticker in STOCKS + ETFS + list(FUTURES):
        row = run_symbol(ticker)
        if row:
            results.append(row)
    if not results:
        print("No symbols produced a held-out session.", file=sys.stderr)
        return 1
    total = sum(row["profit"] for row in results)
    print(f"\nHeld-out profit across these sized positions: ${total:.2f}")
    print("Futures use one micro contract. Stocks and ETFs use 10 shares.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
