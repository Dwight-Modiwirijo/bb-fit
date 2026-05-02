namespace BbFit.Orchestrator.Models;

public record OhlcCandle(
    decimal Open,
    decimal High,
    decimal Low,
    decimal Close,
    decimal Volume,
    DateTimeOffset Timestamp
);
