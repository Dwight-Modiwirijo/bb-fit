using System.Collections.Concurrent;
using System.Text.Json;
using BbFit.Orchestrator.Hubs;
using BbFit.Orchestrator.Models;
using Microsoft.AspNetCore.SignalR;
using Websocket.Client;

namespace BbFit.Orchestrator.Services;

public sealed class KrakenWebSocketClient : BackgroundService
{
    private const string WsUrl = "wss://ws.kraken.com/v2";
    private const float  FeePerSide = 0.0026f;
    private const int    SeqLen     = 64;
    private const int    NumFeatures = 33;

    // z-score normalization stats (t000_ features, training distribution)
    private static readonly float[] Means = [
        0.002730f,              // 0  canonicalFee
        5.000000f,              // 1  intervalMinutes
        5.000000f,              // 2  observedIntervalMinutes
        6305.033927f,           // 3  signalOpen
        6317.415639f,           // 4  signalHigh
        6291.941350f,           // 5  signalLow
        6305.110881f,           // 6  signalClose
        6305.164301f,           // 7  executionOpen
        6317.473335f,           // 8  executionHigh
        6291.934315f,           // 9  executionLow
        6305.148071f,           // 10 executionClose
        6305.164301f,           // 11 executionPrice
        -11.781344f,            // 12 tradeActionRaw
        0.002464f,              // 13 tradeSide
        1.031292f,              // 14 lastTrade
        0.002464f,              // 15 actionTaken
        216007967545.607147f,   // 16 tradingCapital
        31222547.444635f,       // 17 assetsHeld
        0.773910f,              // 18 inPosition
        5832.553092f,           // 19 entryPrice
        1112831261917.468506f,  // 20 positionValue
        1328839229463.061035f,  // 21 netEquity
        3.157993f,              // 22 buyCount
        2.384083f,              // 23 sellCount
        1.003120f,              // 24 wins
        1.380962f,              // 25 losses
        7499908669130.564453f,  // 26 totalTradedNotional
        0.002730f,              // 27 feePerSide
        13499837034.533480f,    // 28 cost
        0.422614f,              // 29 runGroup_code
        0.422614f,              // 30 sourceFile_code
        0.000000f,              // 31 interval_code
        1.000000f,              // 32 splitHint_code
    ];

    private static readonly float[] Stds = [
        0.001087f,
        1.000000f,
        1.000000f,
        11762.180624f,
        11783.696159f,
        11740.132460f,
        11762.540900f,
        11762.601559f,
        11784.057070f,
        11740.082482f,
        11762.746312f,
        11762.601559f,
        2766.062911f,
        0.206946f,
        0.474458f,
        0.206946f,
        2079187742025.969482f,
        107107904.706700f,
        0.418298f,
        11501.837891f,
        4920436873714.497070f,
        5296504494153.166016f,
        3.890849f,
        3.925436f,
        1.591838f,
        2.886772f,
        36069641293895.851562f,
        0.001087f,
        64925354031.654709f,
        0.493975f,
        0.493975f,
        1.000000f,
        1.000000f,
    ];

    private readonly BbFitInferenceService     _inference;
    private readonly OrderManager              _orders;
    private readonly IHubContext<StateHub>     _hub;
    private readonly ILogger<KrakenWebSocketClient> _log;
    private readonly string                    _symbol;
    private readonly int                       _intervalMinutes;

    // Circular buffer of completed OHLC candles
    private readonly ConcurrentQueue<OhlcCandle> _candles = new();
    private OhlcCandle? _inProgressCandle;

    public KrakenWebSocketClient(
        BbFitInferenceService inference,
        OrderManager orders,
        IHubContext<StateHub> hub,
        IConfiguration config,
        ILogger<KrakenWebSocketClient> log)
    {
        _inference       = inference;
        _orders          = orders;
        _hub             = hub;
        _log             = log;
        _symbol          = config["Kraken:TradingPair"] ?? "BTC/EUR";
        _intervalMinutes = config.GetValue<int>("Kraken:OhlcIntervalMinutes", 5);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var uri = new Uri(WsUrl);

        using var client = new WebsocketClient(uri)
        {
            ReconnectTimeout         = TimeSpan.FromSeconds(30),
            ErrorReconnectTimeout    = TimeSpan.FromSeconds(10),
            IsReconnectionEnabled    = true,
        };

        client.ReconnectionHappened.Subscribe(info =>
            _log.LogInformation("WebSocket (her)verbonden: {Type}", info.Type));

        client.DisconnectionHappened.Subscribe(info =>
            _log.LogWarning("WebSocket verbroken: {Type}", info.Type));

        client.MessageReceived.Subscribe(msg => HandleMessage(msg.Text));

        await client.Start();
        _log.LogInformation("Verbonden met Kraken WS v2, handelspaar={Symbol}", _symbol);

        SubscribeOhlc(client);

        await Task.Delay(Timeout.Infinite, stoppingToken);
    }

    private void SubscribeOhlc(WebsocketClient client)
    {
        var subscribe = $$"""
            {
              "method": "subscribe",
              "params": {
                "channel": "ohlc",
                "symbol": ["{{_symbol}}"],
                "interval": {{_intervalMinutes}}
              }
            }
            """;
        client.Send(subscribe);
        _log.LogInformation("OHLC abonnement aangevraagd ({Interval}m)", _intervalMinutes);
    }

    private void HandleMessage(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;

        try
        {
            using var doc = JsonDocument.Parse(text);
            var root = doc.RootElement;

            if (!root.TryGetProperty("channel", out var channelEl)) return;
            if (channelEl.GetString() != "ohlc") return;

            if (!root.TryGetProperty("type", out var typeEl)) return;
            var type = typeEl.GetString();

            if (!root.TryGetProperty("data", out var dataEl)) return;

            foreach (var item in dataEl.EnumerateArray())
                ProcessOhlcItem(item, type == "snapshot");
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Fout bij verwerken WS bericht");
        }
    }

    private void ProcessOhlcItem(JsonElement item, bool isSnapshot)
    {
        var open      = item.GetProperty("open").GetDecimal();
        var high      = item.GetProperty("high").GetDecimal();
        var low       = item.GetProperty("low").GetDecimal();
        var close     = item.GetProperty("close").GetDecimal();
        var volume    = item.TryGetProperty("volume", out var v) ? v.GetDecimal() : 0m;
        var timestamp = item.GetProperty("timestamp").GetDateTimeOffset();

        var candle = new OhlcCandle(open, high, low, close, volume, timestamp);

        if (isSnapshot)
        {
            // Snapshot: vul de buffer met historische kaarsen
            _candles.Enqueue(candle);
            while (_candles.Count > SeqLen)
                _candles.TryDequeue(out _);
            _inProgressCandle = candle;
            return;
        }

        // Live update: detecteer voltooide kaars via timestamp wijziging
        if (_inProgressCandle != null && candle.Timestamp != _inProgressCandle.Timestamp)
        {
            // Vorige kaars is voltooid, zet hem in de buffer
            _candles.Enqueue(_inProgressCandle);
            while (_candles.Count > SeqLen)
                _candles.TryDequeue(out _);

            _log.LogDebug("Kaars voltooid: {Time} O={Open} H={High} L={Low} C={Close}",
                _inProgressCandle.Timestamp, _inProgressCandle.Open,
                _inProgressCandle.High, _inProgressCandle.Low, _inProgressCandle.Close);

            if (_candles.Count == SeqLen)
                RunInference(_inProgressCandle.Close);
        }

        _inProgressCandle = candle;
    }

    private void RunInference(decimal currentClose)
    {
        try
        {
            var candles  = _candles.ToArray();
            var snapshot = _orders.Portfolio.Snapshot(currentClose);
            var features = BuildFeatureVector(candles, snapshot, currentClose);

            var signal = _inference.Predict(features);
            _orders.Portfolio.SetLastSignal(signal);

            _log.LogInformation(
                "Signaal: {Action} ({Confidence:P1}) | Side={Side} | ΔEquity={Delta:F4}",
                signal.Action, signal.Confidence, signal.Side, signal.NetEquityDelta);

            _ = _hub.Clients.All.SendAsync("Signal", signal);
            _ = _hub.Clients.All.SendAsync("Portfolio", _orders.Portfolio.Snapshot(currentClose));

            _orders.HandleSignalAsync(signal, currentClose).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Fout in inferentie/orderbeheer");
        }
    }

    private static float[] BuildFeatureVector(
        OhlcCandle[] candles,
        PortfolioSnapshot snap,
        decimal currentClose)
    {
        var features = new float[SeqLen * NumFeatures];

        for (int t = 0; t < SeqLen; t++)
        {
            var c   = candles[t];
            var idx = t * NumFeatures;
            var p   = (float)currentClose;

            float[] raw = [
                FeePerSide,                          // 0  canonicalFee
                5f,                                  // 1  intervalMinutes
                5f,                                  // 2  observedIntervalMinutes
                (float)c.Open,                       // 3  signalOpen
                (float)c.High,                       // 4  signalHigh
                (float)c.Low,                        // 5  signalLow
                (float)c.Close,                      // 6  signalClose
                (float)c.Open,                       // 7  executionOpen
                (float)c.High,                       // 8  executionHigh
                (float)c.Low,                        // 9  executionLow
                (float)c.Close,                      // 10 executionClose
                (float)c.Open,                       // 11 executionPrice (executed at open)
                0f,                                  // 12 tradeActionRaw
                (float)(snap.InPosition ? 1 : 0),   // 13 tradeSide
                1f,                                  // 14 lastTrade (neutral=1)
                0f,                                  // 15 actionTaken
                (float)snap.TradingCapital,          // 16 tradingCapital
                (float)snap.AssetsHeld,              // 17 assetsHeld
                snap.InPosition ? 1f : 0f,           // 18 inPosition
                (float)snap.EntryPrice,              // 19 entryPrice
                (float)snap.PositionValue,           // 20 positionValue
                (float)snap.NetEquity,               // 21 netEquity
                snap.BuyCount,                       // 22 buyCount
                snap.SellCount,                      // 23 sellCount
                snap.Wins,                           // 24 wins
                snap.Losses,                         // 25 losses
                (float)snap.TotalNotional,           // 26 totalTradedNotional
                FeePerSide,                          // 27 feePerSide
                0f,                                  // 28 cost
                0f,                                  // 29 runGroup_code
                0f,                                  // 30 sourceFile_code
                0f,                                  // 31 interval_code (5min)
                1f,                                  // 32 splitHint_code (live)
            ];

            for (int f = 0; f < NumFeatures; f++)
            {
                float std = Stds[f] > 0f ? Stds[f] : 1f;
                features[idx + f] = (raw[f] - Means[f]) / std;
            }
        }

        return features;
    }
}
