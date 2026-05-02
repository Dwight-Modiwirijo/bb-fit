namespace BbFit.Orchestrator.Models;

public enum TradeAction { Hold = 0, Buy = 1, Sell = 2 }
public enum TradeSide { None = 0, Long = 1, Short = 2 }

public record TradingSignal(
    TradeAction Action,
    TradeSide Side,
    float NetEquityDelta,
    float Confidence,
    DateTimeOffset Timestamp
);
