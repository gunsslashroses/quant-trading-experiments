"""Backtesting engine for evaluating trading strategies."""

from dataclasses import dataclass

from quant_trading.strategy import TradeSignal


@dataclass
class BacktestResult:
    initial_capital: float
    final_value: float
    total_return_pct: float
    num_trades: int
    trades: list[dict]


def run_backtest(
    signals: list[TradeSignal],
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Run a simple backtest given a list of trade signals.

    Buys/sells the full position on each signal.
    """
    cash = initial_capital
    shares = 0.0
    trades: list[dict] = []

    for signal in signals:
        if signal.action == "BUY" and cash > 0:
            shares = cash / signal.price
            trades.append(
                {
                    "timestamp": signal.timestamp,
                    "action": "BUY",
                    "price": signal.price,
                    "shares": shares,
                    "cash_after": 0.0,
                }
            )
            cash = 0.0

        elif signal.action == "SELL" and shares > 0:
            cash = shares * signal.price
            trades.append(
                {
                    "timestamp": signal.timestamp,
                    "action": "SELL",
                    "price": signal.price,
                    "shares": shares,
                    "cash_after": cash,
                }
            )
            shares = 0.0

    final_value = cash if shares == 0 else shares * signals[-1].price if signals else cash
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100

    return BacktestResult(
        initial_capital=initial_capital,
        final_value=round(final_value, 2),
        total_return_pct=round(total_return_pct, 2),
        num_trades=len(trades),
        trades=trades,
    )
