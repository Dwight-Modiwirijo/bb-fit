using BbFit.Orchestrator.Models;
using BbFit.Orchestrator.Services;
using BbFit.Orchestrator.Tests.Helpers;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;

namespace BbFit.Orchestrator.Tests;

public sealed class OrderManagerTests
{
    // ------------------------------------------------------------------ helpers

    private static IConfiguration BuildConfig(
        string apiKey    = "test-key",
        // base64("testsecret") — geldig voor HMAC-SHA512
        string apiSecret = "dGVzdHNlY3JldA==",
        decimal maxPos   = 0.001m,
        decimal stopLoss = 0.02m,
        decimal capital  = 10_000m) =>
        new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Kraken:ApiKey"]          = apiKey,
                ["Kraken:ApiSecret"]       = apiSecret,
                ["Kraken:TradingPair"]     = "BTC/EUR",
                ["Kraken:MaxPositionSize"] = maxPos.ToString("G"),
                ["Kraken:StopLossPct"]     = stopLoss.ToString("G"),
                ["Kraken:InitialCapital"]  = capital.ToString("G"),
            })
            .Build();

    private static (OrderManager Manager, FakeKrakenHandler Handler) Build(
        IConfiguration? config = null)
    {
        var handler = new FakeKrakenHandler();
        var mgr     = new OrderManager(
            config ?? BuildConfig(),
            NullLogger<OrderManager>.Instance,
            handler);
        return (mgr, handler);
    }

    private const string AddOrderOk =
        """{"error":[],"result":{"descr":{"order":"buy 0.001 XBTZEUR @ market"},"txid":["OABC-123456"]}}""";

    private const string AddOrderError =
        """{"error":["EOrder:Insufficient funds"]}""";

    private const string CancelOrderOk =
        """{"error":[],"result":{"count":1,"pending":false}}""";

    private const string CancelOrderError =
        """{"error":["EOrder:Unknown order"]}""";

    private const string OpenOrdersOk =
        """{"error":[],"result":{"open":{"OABC-123456":{"status":"open","vol":"0.001"}}}}""";

    // ------------------------------------------------------------------ PlaceOrderAsync

    [Fact]
    public async Task PlaceOrderAsync_Buy_ReturnsTransactionId()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        var txid = await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);

        Assert.Equal("OABC-123456", txid);
    }

    [Fact]
    public async Task PlaceOrderAsync_Buy_UpdatesPortfolioState()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);

        var snap = mgr.Portfolio.Snapshot(50_000m);
        Assert.True(snap.InPosition);
        Assert.Equal(0.001m, snap.AssetsHeld);
        Assert.Equal(1, snap.BuyCount);
        Assert.Equal(50_000m, snap.EntryPrice);
    }

    [Fact]
    public async Task PlaceOrderAsync_Sell_UpdatesPortfolioAndCountsWin()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy",  0.001m, 50_000m);
        await mgr.PlaceOrderAsync("sell", 0.001m, 55_000m);

        var snap = mgr.Portfolio.Snapshot(55_000m);
        Assert.False(snap.InPosition);
        Assert.Equal(0m, snap.AssetsHeld);
        Assert.Equal(1, snap.Wins);
        Assert.Equal(0, snap.Losses);
    }

    [Fact]
    public async Task PlaceOrderAsync_KrakenReturnsError_ReturnsNullAndDoesNotUpdatePortfolio()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderError);

        var txid = await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);

        Assert.Null(txid);
        Assert.False(mgr.Portfolio.InPosition);
    }

    [Fact]
    public async Task PlaceOrderAsync_MarketOrder_RequestBodyContainsRequiredFields()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);

        var body = handler.Calls[0].Body;
        Assert.Contains("nonce=",        body);
        Assert.Contains("type=buy",      body);
        Assert.Contains("ordertype=market", body);
        Assert.Contains("volume=",       body);
        Assert.Contains("pair=XBTZEUR", body);
        Assert.DoesNotContain("price=",  body); // geen prijs bij market order
    }

    [Fact]
    public async Task PlaceOrderAsync_LimitOrder_RequestBodyContainsPrice()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m, orderType: "limit");

        var body = handler.Calls[0].Body;
        Assert.Contains("ordertype=limit", body);
        Assert.Contains("price=",          body);
    }

    [Fact]
    public async Task PlaceOrderAsync_RequestHasApiSignHeader()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);

        Assert.True(handler.Calls[0].HasApiSign);
    }

    // ------------------------------------------------------------------ CancelOrderAsync

    [Fact]
    public async Task CancelOrderAsync_Success_ReturnsTrue()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("CancelOrder", CancelOrderOk);

        var result = await mgr.CancelOrderAsync("OABC-123456");

        Assert.True(result);
        Assert.Single(handler.Calls);
        Assert.Contains("CancelOrder", handler.Calls[0].Path);
    }

    [Fact]
    public async Task CancelOrderAsync_KrakenError_ReturnsFalse()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("CancelOrder", CancelOrderError);

        var result = await mgr.CancelOrderAsync("UNKNOWN-TX");

        Assert.False(result);
    }

    [Fact]
    public async Task CancelOrderAsync_RequestBodyContainsTxid()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("CancelOrder", CancelOrderOk);

        await mgr.CancelOrderAsync("OABC-123456");

        Assert.Contains("txid=OABC-123456", handler.Calls[0].Body);
    }

    // ------------------------------------------------------------------ GetOpenOrdersAsync

    [Fact]
    public async Task GetOpenOrdersAsync_ReturnsOpenOrdersElement()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("OpenOrders", OpenOrdersOk);

        var result = await mgr.GetOpenOrdersAsync();

        Assert.NotNull(result);
        Assert.True(result.Value.TryGetProperty("open", out _));
    }

    [Fact]
    public async Task GetOpenOrdersAsync_HttpError_ReturnsNull()
    {
        var (mgr, handler) = Build();
        // geen route → 404

        var result = await mgr.GetOpenOrdersAsync();

        Assert.Null(result);
    }

    // ------------------------------------------------------------------ HandleSignalAsync

    [Fact]
    public async Task HandleSignalAsync_Hold_DoesNotCallKraken()
    {
        var (mgr, handler) = Build();
        var signal = new TradingSignal(TradeAction.Hold, TradeSide.None, 0f, 0.9f, DateTimeOffset.UtcNow);

        await mgr.HandleSignalAsync(signal, 50_000m);

        Assert.Empty(handler.Calls);
    }

    [Fact]
    public async Task HandleSignalAsync_Buy_NotInPosition_PlacesBuyOrder()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);
        var signal = new TradingSignal(TradeAction.Buy, TradeSide.Long, 0.1f, 0.9f, DateTimeOffset.UtcNow);

        await mgr.HandleSignalAsync(signal, 50_000m);

        Assert.Single(handler.Calls);
        Assert.Contains("type=buy", handler.Calls[0].Body);
    }

    [Fact]
    public async Task HandleSignalAsync_Buy_AlreadyInPosition_SkipsOrder()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        // Positie openen
        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);
        handler.Calls.Clear();

        var signal = new TradingSignal(TradeAction.Buy, TradeSide.Long, 0.1f, 0.9f, DateTimeOffset.UtcNow);
        await mgr.HandleSignalAsync(signal, 50_000m);

        Assert.Empty(handler.Calls);
    }

    [Fact]
    public async Task HandleSignalAsync_Sell_InPosition_PlacesSellOrder()
    {
        var (mgr, handler) = Build();
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);
        handler.Calls.Clear();

        var signal = new TradingSignal(TradeAction.Sell, TradeSide.None, -0.1f, 0.8f, DateTimeOffset.UtcNow);
        await mgr.HandleSignalAsync(signal, 55_000m);

        Assert.Single(handler.Calls);
        Assert.Contains("type=sell", handler.Calls[0].Body);
    }

    [Fact]
    public async Task HandleSignalAsync_Sell_NotInPosition_SkipsOrder()
    {
        var (mgr, handler) = Build();
        var signal = new TradingSignal(TradeAction.Sell, TradeSide.None, -0.1f, 0.8f, DateTimeOffset.UtcNow);

        await mgr.HandleSignalAsync(signal, 50_000m);

        Assert.Empty(handler.Calls);
    }

    [Fact]
    public async Task HandleSignalAsync_StopLoss_TriggersWhenLossReachesThreshold()
    {
        // Stop-loss bij 2%; koop op 50_000, stuur Buy-signaal op 49_000 (2% verlies)
        // Het Buy-signaal wordt genegeerd (al in positie), maar stop-loss triggert wel.
        var (mgr, handler) = Build(BuildConfig(stopLoss: 0.02m));
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);
        handler.Calls.Clear();

        var signal = new TradingSignal(TradeAction.Buy, TradeSide.Long, 0f, 0.5f, DateTimeOffset.UtcNow);
        await mgr.HandleSignalAsync(signal, 49_000m); // verlies = (50k-49k)/50k = 0.02 ≥ 0.02

        Assert.Single(handler.Calls);
        Assert.Contains("type=sell", handler.Calls[0].Body);
    }

    [Fact]
    public async Task HandleSignalAsync_StopLoss_DoesNotTrigger_BelowThreshold()
    {
        var (mgr, handler) = Build(BuildConfig(stopLoss: 0.02m));
        handler.SetResponse("AddOrder", AddOrderOk);

        await mgr.PlaceOrderAsync("buy", 0.001m, 50_000m);
        handler.Calls.Clear();

        var signal = new TradingSignal(TradeAction.Buy, TradeSide.Long, 0f, 0.5f, DateTimeOffset.UtcNow);
        await mgr.HandleSignalAsync(signal, 49_500m); // verlies = 1% < drempel van 2%

        Assert.Empty(handler.Calls);
    }
}
