namespace BbFit.Orchestrator.Models;

public sealed class PortfolioState
{
    private readonly Lock _lock = new();

    // Initieel kapitaal in quote-currency (bijv. EUR)
    public decimal TradingCapital  { get; private set; }
    public decimal AssetsHeld      { get; private set; }
    public bool    InPosition      { get; private set; }
    public decimal EntryPrice      { get; private set; }
    public int     BuyCount        { get; private set; }
    public int     SellCount       { get; private set; }
    public int     Wins            { get; private set; }
    public int     Losses          { get; private set; }
    public decimal TotalNotional   { get; private set; }
    public decimal LastTradePrice  { get; private set; }
    public TradingSignal? LastSignal { get; private set; }

    public PortfolioState(decimal initialCapital)
    {
        TradingCapital = initialCapital;
        LastTradePrice = 0m;
    }

    public decimal NetEquity(decimal currentPrice)
    {
        lock (_lock) return TradingCapital + AssetsHeld * currentPrice;
    }

    public decimal PositionValue(decimal currentPrice)
    {
        lock (_lock) return AssetsHeld * currentPrice;
    }

    public void RecordBuy(decimal price, decimal quantity, decimal fee)
    {
        lock (_lock)
        {
            var cost = price * quantity + fee;
            TradingCapital -= cost;
            AssetsHeld     += quantity;
            InPosition      = true;
            EntryPrice      = price;
            LastTradePrice  = price;
            BuyCount++;
            TotalNotional  += price * quantity;
        }
    }

    public void RecordSell(decimal price, decimal quantity, decimal fee)
    {
        lock (_lock)
        {
            var proceeds = price * quantity - fee;
            TradingCapital += proceeds;
            AssetsHeld     -= quantity;

            if (price > EntryPrice) Wins++; else Losses++;

            InPosition     = AssetsHeld > 0;
            EntryPrice     = InPosition ? EntryPrice : 0m;
            LastTradePrice = price;
            SellCount++;
            TotalNotional += price * quantity;
        }
    }

    public void SetLastSignal(TradingSignal signal)
    {
        lock (_lock) LastSignal = signal;
    }

    public PortfolioSnapshot Snapshot(decimal currentPrice)
    {
        lock (_lock)
        {
            return new PortfolioSnapshot(
                TradingCapital,
                AssetsHeld,
                InPosition,
                EntryPrice,
                PositionValue(currentPrice),
                NetEquity(currentPrice),
                BuyCount, SellCount, Wins, Losses,
                TotalNotional,
                LastSignal
            );
        }
    }
}

public record PortfolioSnapshot(
    decimal TradingCapital,
    decimal AssetsHeld,
    bool    InPosition,
    decimal EntryPrice,
    decimal PositionValue,
    decimal NetEquity,
    int     BuyCount,
    int     SellCount,
    int     Wins,
    int     Losses,
    decimal TotalNotional,
    TradingSignal? LastSignal
);
