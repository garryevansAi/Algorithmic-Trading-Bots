#!/usr/bin/env python3
"""Broker connection for the session simulation.

The default broker is a local paper account. TradeLocker is used only when
TRADELOCKER_EMAIL, TRADELOCKER_PASSWORD, and TRADELOCKER_SERVER are set.
This module lists accounts. It does not place an order unless place_order is called.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEMO_BASE = "https://demo.tradelocker.com/backend-api"
LIVE_BASE = "https://live.tradelocker.com/backend-api"


class BrokerError(RuntimeError):
    pass


class PaperBroker:
    name = "paper"

    def __init__(self, cash: float = 10_000.0) -> None:
        self.cash = cash

    def connect(self) -> dict:
        return {
            "broker": self.name,
            "environment": "local",
            "accounts": [
                {
                    "id": "paper",
                    "name": "Simulated account",
                    "currency": "USD",
                    "accNum": "1",
                    "balance": self.cash,
                }
            ],
        }


class TradeLockerBroker:
    name = "tradelocker"

    def __init__(self, environment: str | None = None) -> None:
        self.environment = (environment or os.environ.get("TRADELOCKER_ENV") or "demo").lower()
        if self.environment not in {"demo", "live"}:
            raise BrokerError("TRADELOCKER_ENV must be demo or live.")
        self.base_url = DEMO_BASE if self.environment == "demo" else LIVE_BASE
        self.email = os.environ.get("TRADELOCKER_EMAIL", "")
        self.password = os.environ.get("TRADELOCKER_PASSWORD", "")
        self.server = os.environ.get("TRADELOCKER_SERVER", "")
        self.account_id = os.environ.get("TRADELOCKER_ACCOUNT_ID", "")
        self._token = ""

    def missing_credentials(self) -> list[str]:
        missing = []
        if not self.email:
            missing.append("TRADELOCKER_EMAIL")
        if not self.password:
            missing.append("TRADELOCKER_PASSWORD")
        if not self.server:
            missing.append("TRADELOCKER_SERVER")
        return missing

    def connect(self) -> dict:
        missing = self.missing_credentials()
        if missing:
            raise BrokerError(
                "TradeLocker credentials are not set: " + ", ".join(missing) + ". "
                "Set them in the environment. TRADELOCKER_SERVER is the server name from the TradeLocker login screen. "
                "TRADELOCKER_ENV defaults to demo."
            )
        tokens = self._request(
            "POST",
            "/auth/jwt/token",
            {"email": self.email, "password": self.password, "server": self.server},
        )
        self._token = str(tokens.get("accessToken", ""))
        if not self._token:
            raise BrokerError("TradeLocker did not return an access token.")
        listed = self._request("GET", "/auth/jwt/all-accounts")
        accounts = []
        for account in listed.get("accounts", []):
            accounts.append(
                {
                    "id": str(account.get("id", "")),
                    "name": account.get("name", ""),
                    "currency": account.get("currency", ""),
                    "accNum": str(account.get("accNum", "")),
                    "balance": account.get("aaccountBalance", account.get("balance")),
                    "status": account.get("status", ""),
                }
            )
        if self.account_id:
            accounts = [account for account in accounts if account["id"] == self.account_id]
            if not accounts:
                raise BrokerError(f"TRADELOCKER_ACCOUNT_ID {self.account_id} was not in the account list.")
        return {"broker": self.name, "environment": self.environment, "accounts": accounts}

    def place_order(self, account_id: str, acc_num: str, instrument_id: int, side: str, quantity: float) -> dict:
        if side not in {"buy", "sell"}:
            raise BrokerError("side must be buy or sell.")
        if quantity <= 0:
            raise BrokerError("quantity must be positive.")
        return self._request(
            "POST",
            f"/trade/accounts/{account_id}/orders",
            {
                "qty": quantity,
                "routeId": "TRADE",
                "side": side,
                "validity": "IOC",
                "type": "market",
                "tradableInstrumentId": instrument_id,
                "price": 0,
            },
            headers={"accNum": acc_num},
        )

    def _request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base_url + path, data=data, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise BrokerError(f"TradeLocker {method} {path} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise BrokerError(f"TradeLocker {method} {path} failed: {exc.reason}") from exc
        if not payload:
            return {}
        return json.loads(payload)


def open_broker(name: str) -> PaperBroker | TradeLockerBroker:
    if name == "paper":
        return PaperBroker()
    if name == "tradelocker":
        return TradeLockerBroker()
    raise BrokerError("broker must be paper or tradelocker.")


def format_connection(info: dict) -> str:
    lines = [f"Broker: {info['broker']} ({info['environment']})"]
    for account in info["accounts"]:
        balance = account.get("balance")
        balance_text = "n/a" if balance is None else f"{balance}"
        lines.append(
            f"  account {account['id']} accNum={account['accNum']} "
            f"{account.get('currency', '')} balance={balance_text} {account.get('name', '')}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the broker connection without placing an order.")
    parser.add_argument("--broker", choices=["paper", "tradelocker"], default="paper")
    args = parser.parse_args()
    try:
        info = open_broker(args.broker).connect()
    except BrokerError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(format_connection(info))
    print("No order was sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
