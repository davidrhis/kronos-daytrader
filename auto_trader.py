#!/usr/bin/env python3
"""
Kronos AI + DayTrader: Autonomous Market Scanner & Groww Execution Engine
Performs scheduled market scanning, Monte Carlo expectancy filtering,
and automated bracket order execution via Groww API with headless TOTP authentication.
"""

import os
import sys
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

# Ensure root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(CURRENT_DIR, ".env"))
except ImportError:
    pass

import pyotp
try:
    from growwapi import GrowwAPI
except ImportError:
    GrowwAPI = None

from engine import run_full_analysis

IST = ZoneInfo("Asia/Kolkata")
HISTORY_DIR = os.path.join(CURRENT_DIR, "history")
AUTO_TRADES_FILE = os.path.join(HISTORY_DIR, "auto_trades.json")
AUTO_STATUS_FILE = os.path.join(HISTORY_DIR, "auto_status.json")
TOKEN_CACHE_FILE = os.path.join(CURRENT_DIR, ".groww_token")

os.makedirs(HISTORY_DIR, exist_ok=True)

class GrowwAuthManager:
    """Manages headless automated Groww API token generation via TOTP."""

    def __init__(self):
        self.api_key = os.getenv("GROWW_API_KEY", "").strip()
        self.totp_secret = os.getenv("GROWW_TOTP_SECRET", "").strip()
        self.access_token = os.getenv("GROWW_ACCESS_TOKEN", "").strip()
        self._load_cached_token()

    def _load_cached_token(self):
        # Also check server's existing groww-dashboard token if present
        fallback_token_path = Path("/home/opc/groww-dashboard/token")
        if not self.access_token and os.path.exists(TOKEN_CACHE_FILE):
            try:
                self.access_token = Path(TOKEN_CACHE_FILE).read_text().strip()
            except Exception:
                pass
        elif not self.access_token and fallback_token_path.exists():
            try:
                self.access_token = fallback_token_path.read_text().strip()
            except Exception:
                pass

    def get_token(self) -> Optional[str]:
        # If token exists, test if valid
        if self.access_token and self._verify_token(self.access_token):
            return self.access_token

        # Attempt automated TOTP renewal if credentials configured
        if self.api_key and self.totp_secret:
            print("🔑 Generating fresh Groww access token via automated TOTP flow...")
            try:
                totp_gen = pyotp.TOTP(self.totp_secret)
                current_otp = totp_gen.now()
                token = GrowwAPI.get_access_token(api_key=self.api_key, totp=current_otp)
                if token:
                    self.access_token = token
                    try:
                        Path(TOKEN_CACHE_FILE).write_text(token)
                    except Exception:
                        pass
                    print("✅ Successfully acquired new Groww access token via TOTP.")
                    return self.access_token
            except Exception as e:
                print(f"⚠️ Automated TOTP token generation failed: {e}")

        return self.access_token if self.access_token else None

    def _verify_token(self, token: str) -> bool:
        import urllib.request
        try:
            req = urllib.request.Request(
                "https://api.groww.in/v1/positions/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "X-API-VERSION": "1.0"
                }
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_client(self) -> Optional[Any]:
        token = self.get_token()
        if token and GrowwAPI:
            try:
                return GrowwAPI(token)
            except Exception as e:
                print(f"⚠️ GrowwAPI client initialization error: {e}")
        return None


class AutoTrader:
    """Autonomous market scanner and risk-managed execution engine."""

    def __init__(self):
        self.auth = GrowwAuthManager()
        self._load_config()

    def _load_config(self):
        self.enabled = os.getenv("AUTO_TRADE_ENABLED", "true").lower() in ["true", "1", "yes"]
        self.dry_run = os.getenv("DRY_RUN", "true").lower() in ["true", "1", "yes"]
        self.capital = float(os.getenv("ACCOUNT_CAPITAL", "100000.0"))
        self.risk_pct = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
        self.min_expectancy = float(os.getenv("MIN_EXPECTANCY_R", "0.35"))
        self.min_target_pct = float(os.getenv("MIN_TARGET_ODDS_PCT", "52.0"))
        self.max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "3"))
        
        default_watchlist = "^NSEI,^NSEBANK,RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,SBIN.NS,KALYANKJIL.NS"
        raw_watchlist = os.getenv("WATCHLIST", default_watchlist)
        self.watchlist = [s.strip().upper() for s in raw_watchlist.split(",") if s.strip()]

    def get_today_trades_count(self) -> int:
        trades = self.load_trades()
        today_str = datetime.datetime.now(IST).strftime("%Y-%m-%d")
        return sum(1 for t in trades if t.get("timestamp", "").startswith(today_str))

    def load_trades(self) -> List[Dict[str, Any]]:
        if os.path.exists(AUTO_TRADES_FILE):
            try:
                with open(AUTO_TRADES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def save_trade(self, trade_record: Dict[str, Any]):
        trades = self.load_trades()
        trades.insert(0, trade_record)
        trades = trades[:100]
        with open(AUTO_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)

    def execute_order(self, ticker: str, brackets: Dict[str, Any], timemachine: Dict[str, Any]) -> Dict[str, Any]:
        """Places real or paper bracket order depending on DRY_RUN mode."""
        shares = brackets.get("recommended_shares", 0)
        entry = brackets.get("entry", 0.0)
        stop = brackets.get("stop_loss", 0.0)
        target = brackets.get("target_2r", 0.0)

        # Normalize symbol for Groww (strip .NS)
        clean_symbol = ticker.replace(".NS", "").replace("^", "")
        now_str = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "timestamp": now_str,
            "ticker": ticker,
            "trading_symbol": clean_symbol,
            "quantity": shares,
            "entry_price": entry,
            "stop_loss": stop,
            "target": target,
            "expectancy_r": timemachine.get("expectancy_r"),
            "target_odds_pct": timemachine.get("target_first_pct"),
            "verdict": timemachine.get("verdict"),
            "mode": "PAPER_TRADING" if self.dry_run else "LIVE_BROKER",
            "status": "EXECUTED" if self.dry_run else "SUBMITTED"
        }

        if self.dry_run:
            print(f"📝 [PAPER TRADE] BUY {shares} shares of {clean_symbol} @ Limit ₹{entry} | Stop ₹{stop} | Target ₹{target}")
            record["order_id"] = f"PAPER-{int(time.time())}"
            self.save_trade(record)
            return record

        # LIVE ORDER EXECUTION via Groww API
        client = self.auth.get_client()
        if not client:
            print("⚠️ Live order failed: No active Groww API client available. Falling back to paper trade.")
            record["mode"] = "PAPER_FALLBACK (NO_CLIENT)"
            record["order_id"] = f"PAPER-{int(time.time())}"
            self.save_trade(record)
            return record

        try:
            print(f"⚡ [LIVE GROWW ORDER] Placing BUY order for {shares} units of {clean_symbol}...")
            resp = client.place_order(
                trading_symbol=clean_symbol,
                quantity=shares,
                validity=client.VALIDITY_DAY,
                exchange=client.EXCHANGE_NSE,
                segment=client.SEGMENT_CASH,
                product=client.PRODUCT_MIS,
                order_type=client.ORDER_TYPE_LIMIT,
                transaction_type=client.TRANSACTION_TYPE_BUY,
                price=entry,
                trigger_price=stop
            )
            order_id = resp.get("order_id", f"GW-{int(time.time())}")
            record["order_id"] = order_id
            record["broker_response"] = resp
            print(f"✅ Groww Order Executed Successfully! Order ID: {order_id}")
            self.save_trade(record)
            return record
        except Exception as e:
            print(f"❌ Groww Order Placement Error: {e}")
            record["status"] = "FAILED"
            record["error"] = str(e)
            self.save_trade(record)
            return record

    def run_scan(self) -> Dict[str, Any]:
        """Runs complete watchlist scan and executes on positive expectancy."""
        now = datetime.datetime.now(IST)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*65}\n🤖 Kronos Autonomous Market Scan Initiated ({now_str} IST)\n{'='*65}")

        self._load_config()
        if not self.enabled:
            print("AutoTrader is currently disabled in configuration.")
            return {"status": "DISABLED", "time": now_str}

        executed_today = self.get_today_trades_count()
        remaining_slots = max(0, self.max_trades_per_day - executed_today)
        print(f"Trades Executed Today: {executed_today} / {self.max_trades_per_day} (Remaining slots: {remaining_slots})")
        print(f"Mode: {'DRY RUN (Paper Trading)' if self.dry_run else 'LIVE ORDER EXECUTION (Groww API)'}")
        print(f"Watchlist: {len(self.watchlist)} assets | Min Expectancy: +{self.min_expectancy}R | Min Target Odds: {self.min_target_pct}%")

        signals = []
        executed_orders = []

        for ticker in self.watchlist:
            # Skip indices from equity share purchase unless ETF/Future
            if ticker.startswith("^") and not ticker.endswith(".NS"):
                continue

            if remaining_slots <= 0:
                print(f"Daily trade limit ({self.max_trades_per_day}) reached. Halting scan.")
                break

            try:
                print(f"\nAnalyzing {ticker}...")
                analysis = run_full_analysis(
                    ticker=ticker,
                    timeframe="daily",
                    model_name="small",
                    capital=self.capital,
                    risk_pct=self.risk_pct,
                    paths=10
                )
                tm = analysis.get("timemachine", {})
                brackets = analysis.get("brackets", {})

                exp_r = tm.get("expectancy_r", 0.0)
                tgt_pct = tm.get("target_first_pct", 0.0)
                shares = brackets.get("recommended_shares", 0)

                qualifies = (exp_r >= self.min_expectancy) and (tgt_pct >= self.min_target_pct) and (shares > 0)
                status_icon = "🎯 QUALIFIED" if qualifies else "⏭️ SKIPPED"
                print(f"[{ticker}] Target Odds: {tgt_pct}% | Expectancy: {exp_r:+.2f}R | Shares: {shares} -> {status_icon}")

                signals.append({
                    "ticker": ticker,
                    "target_odds": tgt_pct,
                    "expectancy_r": exp_r,
                    "shares": shares,
                    "qualifies": qualifies
                })

                if qualifies:
                    order_result = self.execute_order(ticker, brackets, tm)
                    executed_orders.append(order_result)
                    remaining_slots -= 1

            except Exception as e:
                print(f"⚠️ Error evaluating {ticker}: {e}")

        summary = {
            "timestamp": now_str,
            "scanned": len(signals),
            "qualified": len(executed_orders),
            "orders": executed_orders,
            "mode": "DRY_RUN" if self.dry_run else "LIVE"
        }

        # Save latest status
        with open(AUTO_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*65}\nScan Finished. Executed {len(executed_orders)} orders.\n{'='*65}\n")
        return summary


if __name__ == "__main__":
    trader = AutoTrader()
    trader.run_scan()
