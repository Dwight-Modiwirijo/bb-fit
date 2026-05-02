using BbFit.Orchestrator.Models;

namespace BbFit.Orchestrator.Tests;

public sealed class PortfolioStateTests
{
    [Fact]
    public void InitialState_IsCorrect()
    {
        var state = new PortfolioState(1_000m);

        Assert.Equal(1_000m, state.TradingCapital);
        Assert.Equal(0m, state.AssetsHeld);
        Assert.False(state.InPosition);
        Assert.Equal(0m, state.EntryPrice);
        Assert.Equal(0, state.BuyCount);
        Assert.Equal(0, state.SellCount);
        Assert.Equal(0, state.Wins);
        Assert.Equal(0, state.Losses);
    }

    [Fact]
    public void RecordBuy_DeductsCapital_AndSetsPosition()
    {
        var state = new PortfolioState(10_000m);
        var fee   = 0.001m * 50_000m * 0.0026m; // 0.13

        state.RecordBuy(price: 50_000m, quantity: 0.001m, fee: fee);

        var expectedCapital = 10_000m - (50_000m * 0.001m + fee);
        Assert.Equal(expectedCapital, state.TradingCapital);
        Assert.Equal(0.001m, state.AssetsHeld);
        Assert.True(state.InPosition);
        Assert.Equal(50_000m, state.EntryPrice);
        Assert.Equal(1, state.BuyCount);
        Assert.Equal(50_000m * 0.001m, state.TotalNotional);
    }

    [Fact]
    public void RecordSell_AtProfit_CountsWin_AndClearsPosition()
    {
        var state = new PortfolioState(10_000m);
        state.RecordBuy(50_000m, 0.001m, 0.13m);
        state.RecordSell(55_000m, 0.001m, 0.143m);

        Assert.Equal(1, state.Wins);
        Assert.Equal(0, state.Losses);
        Assert.Equal(1, state.SellCount);
        Assert.False(state.InPosition);
        Assert.Equal(0m, state.AssetsHeld);
        Assert.Equal(0m, state.EntryPrice);
    }

    [Fact]
    public void RecordSell_AtLoss_CountsLoss()
    {
        var state = new PortfolioState(10_000m);
        state.RecordBuy(50_000m, 0.001m, 0.13m);
        state.RecordSell(45_000m, 0.001m, 0.117m);

        Assert.Equal(0, state.Wins);
        Assert.Equal(1, state.Losses);
    }

    [Fact]
    public void RecordSell_AtSamePrice_CountsLoss()
    {
        // gelijke prijs → geen winst, dus verlies
        var state = new PortfolioState(10_000m);
        state.RecordBuy(50_000m, 0.001m, 0.13m);
        state.RecordSell(50_000m, 0.001m, 0.13m);

        Assert.Equal(0, state.Wins);
        Assert.Equal(1, state.Losses);
    }

    [Fact]
    public void RecordSell_PartialPosition_KeepsInPosition()
    {
        var state = new PortfolioState(10_000m);
        state.RecordBuy(50_000m, 0.002m, 0.26m);
        state.RecordSell(50_000m, 0.001m, 0.13m);

        Assert.True(state.InPosition);
        Assert.Equal(0.001m, state.AssetsHeld);
    }

    [Fact]
    public void NetEquity_EqualsCapitalPlusPositionValue()
    {
        var state = new PortfolioState(10_000m);
        state.RecordBuy(50_000m, 0.001m, 0.13m);

        var expectedCapital  = 10_000m - (50_000m * 0.001m + 0.13m);
        var expectedEquity   = expectedCapital + 0.001m * 60_000m;

        Assert.Equal(expectedEquity, state.NetEquity(60_000m));
    }

    [Fact]
    public void PositionValue_IsZero_WhenNoPosition()
    {
        var state = new PortfolioState(5_000m);
        Assert.Equal(0m, state.PositionValue(99_000m));
    }

    [Fact]
    public void Snapshot_ReflectsCurrentState()
    {
        var state = new PortfolioState(5_000m);
        state.RecordBuy(40_000m, 0.001m, 0.104m);

        var snap = state.Snapshot(42_000m);

        Assert.Equal(state.TradingCapital, snap.TradingCapital);
        Assert.Equal(state.AssetsHeld,     snap.AssetsHeld);
        Assert.True(snap.InPosition);
        Assert.Equal(40_000m, snap.EntryPrice);
        Assert.Equal(0.001m * 42_000m, snap.PositionValue);
        Assert.Equal(state.NetEquity(42_000m), snap.NetEquity);
        Assert.Equal(1, snap.BuyCount);
    }

    [Fact]
    public void MultipleBuySell_AccumulatesCounters()
    {
        var state = new PortfolioState(100_000m);

        state.RecordBuy(50_000m,  0.01m, 1.30m);
        state.RecordSell(55_000m, 0.01m, 1.43m); // win
        state.RecordBuy(55_000m,  0.01m, 1.43m);
        state.RecordSell(50_000m, 0.01m, 1.30m); // loss

        Assert.Equal(2, state.BuyCount);
        Assert.Equal(2, state.SellCount);
        Assert.Equal(1, state.Wins);
        Assert.Equal(1, state.Losses);
    }

    [Fact]
    public void TotalNotional_AccumulatesAcrossTrades()
    {
        var state = new PortfolioState(10_000m);
        state.RecordBuy(50_000m,  0.001m, 0.13m);
        state.RecordSell(55_000m, 0.001m, 0.143m);

        Assert.Equal(50_000m * 0.001m + 55_000m * 0.001m, state.TotalNotional);
    }
}
