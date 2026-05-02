using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using BbFit.Orchestrator.Models;

namespace BbFit.Orchestrator.Services;

public sealed class OrderManager
{
    private const string BaseUrl = "https://api.kraken.com";

    private readonly HttpClient _http;
    private readonly string     _apiKey;
    private readonly string     _apiSecret;
    private readonly decimal    _maxPositionSize;
    private readonly decimal    _stopLossPct;
    private readonly ILogger<OrderManager> _log;

    public PortfolioState Portfolio { get; }

    public OrderManager(IConfiguration config, ILogger<OrderManager> log)
        : this(config, log, new SocketsHttpHandler()) { }

    internal OrderManager(IConfiguration config, ILogger<OrderManager> log, HttpMessageHandler handler)
    {
        _log             = log;
        _apiKey          = config["Kraken:ApiKey"]    ?? string.Empty;
        _apiSecret       = config["Kraken:ApiSecret"] ?? string.Empty;
        _maxPositionSize = config.GetValue<decimal>("Kraken:MaxPositionSize", 0.01m);
        _stopLossPct     = config.GetValue<decimal>("Kraken:StopLossPct", 0.02m);

        var initialCapital = config.GetValue<decimal>("Kraken:InitialCapital", 1000m);
        Portfolio = new PortfolioState(initialCapital);

        _http = new HttpClient(handler) { BaseAddress = new Uri(BaseUrl) };
        if (!string.IsNullOrEmpty(_apiKey))
            _http.DefaultRequestHeaders.Add("API-Key", _apiKey);
    }

    public async Task HandleSignalAsync(TradingSignal signal, decimal currentPrice)
    {
        if (signal.Action == TradeAction.Hold) return;

        var snap = Portfolio.Snapshot(currentPrice);

        if (signal.Action == TradeAction.Buy && !snap.InPosition)
        {
            var quantity = _maxPositionSize;
            _log.LogInformation("BUY {Qty} @ ~{Price}", quantity, currentPrice);
            await PlaceOrderAsync("buy", quantity, currentPrice);
        }
        else if (signal.Action == TradeAction.Sell && snap.InPosition)
        {
            var quantity = snap.AssetsHeld;
            _log.LogInformation("SELL {Qty} @ ~{Price}", quantity, currentPrice);
            await PlaceOrderAsync("sell", quantity, currentPrice);
        }

        // Stop-loss check
        if (snap.InPosition && snap.EntryPrice > 0)
        {
            var loss = (snap.EntryPrice - currentPrice) / snap.EntryPrice;
            if (loss >= _stopLossPct)
            {
                _log.LogWarning("STOP-LOSS geraakt ({Loss:P2}), positie sluiten", loss);
                await PlaceOrderAsync("sell", snap.AssetsHeld, currentPrice);
            }
        }
    }

    public async Task<string?> PlaceOrderAsync(
        string side,
        decimal volume,
        decimal limitPrice,
        string orderType = "market",
        CancellationToken ct = default)
    {
        var postData = new Dictionary<string, string>
        {
            ["nonce"]      = Nonce(),
            ["ordertype"]  = orderType,
            ["type"]       = side,
            ["volume"]     = volume.ToString("G"),
            ["pair"]       = "XBTZEUR",
        };

        if (orderType == "limit")
            postData["price"] = limitPrice.ToString("F2");

        var result = await PostPrivateAsync("/0/private/AddOrder", postData, ct);
        if (result is null) return null;

        using var doc = JsonDocument.Parse(result);
        var root   = doc.RootElement;
        var errors = root.GetProperty("error").EnumerateArray().ToArray();
        if (errors.Length > 0)
        {
            _log.LogError("Kraken orderfout: {Errors}", string.Join(", ", errors.Select(e => e.GetString())));
            return null;
        }

        var txid = root.GetProperty("result")
                       .GetProperty("txid")
                       .EnumerateArray()
                       .FirstOrDefault()
                       .GetString();

        _log.LogInformation("Order geplaatst: {Side} {Volume} txid={Txid}", side, volume, txid);

        decimal fee = volume * limitPrice * 0.0026m;
        if (side == "buy")
            Portfolio.RecordBuy(limitPrice, volume, fee);
        else
            Portfolio.RecordSell(limitPrice, volume, fee);

        return txid;
    }

    public async Task<bool> CancelOrderAsync(string txid, CancellationToken ct = default)
    {
        var postData = new Dictionary<string, string>
        {
            ["nonce"] = Nonce(),
            ["txid"]  = txid,
        };

        var result = await PostPrivateAsync("/0/private/CancelOrder", postData, ct);
        if (result is null) return false;

        using var doc  = JsonDocument.Parse(result);
        var errors = doc.RootElement.GetProperty("error").EnumerateArray().ToArray();
        if (errors.Length > 0)
        {
            _log.LogError("Annuleer fout: {Errors}", string.Join(", ", errors.Select(e => e.GetString())));
            return false;
        }

        _log.LogInformation("Order geannuleerd: {Txid}", txid);
        return true;
    }

    public async Task<JsonElement?> GetOpenOrdersAsync(CancellationToken ct = default)
    {
        var postData = new Dictionary<string, string>
        {
            ["nonce"] = Nonce(),
        };

        var result = await PostPrivateAsync("/0/private/OpenOrders", postData, ct);
        if (result is null) return null;

        using var doc = JsonDocument.Parse(result);
        return doc.RootElement.GetProperty("result").Clone();
    }

    private async Task<string?> PostPrivateAsync(
        string path,
        Dictionary<string, string> data,
        CancellationToken ct)
    {
        var nonce    = data["nonce"];
        var encoded  = string.Join("&", data.Select(kv => $"{kv.Key}={Uri.EscapeDataString(kv.Value)}"));
        var sign     = Sign(path, nonce, encoded);

        var request = new HttpRequestMessage(HttpMethod.Post, path)
        {
            Content = new StringContent(encoded, Encoding.UTF8, "application/x-www-form-urlencoded"),
        };
        request.Headers.Add("API-Sign", sign);

        try
        {
            var response = await _http.SendAsync(request, ct);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync(ct);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "HTTP fout bij {Path}", path);
            return null;
        }
    }

    private string Sign(string path, string nonce, string postData)
    {
        // SHA256(nonce + postData)
        var sha256 = SHA256.HashData(Encoding.UTF8.GetBytes(nonce + postData));

        // HMAC-SHA512(path_bytes + sha256_bytes) met base64-decoded secret
        var secretBytes = Convert.FromBase64String(_apiSecret);
        var pathBytes   = Encoding.UTF8.GetBytes(path);
        var message     = pathBytes.Concat(sha256).ToArray();

        using var hmac = new HMACSHA512(secretBytes);
        return Convert.ToBase64String(hmac.ComputeHash(message));
    }

    private static string Nonce() =>
        DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString();
}
