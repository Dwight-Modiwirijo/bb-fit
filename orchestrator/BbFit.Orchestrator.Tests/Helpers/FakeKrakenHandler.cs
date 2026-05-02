using System.Net;
using System.Text;

namespace BbFit.Orchestrator.Tests.Helpers;

/// <summary>
/// Vervangt de echte Kraken HTTP-verbinding in integratie tests.
/// Registreer responses per padgedeelte; elke aanroep naar dat pad
/// retourneert dezelfde JSON (herbruikbaar voor meerdere aanroepen).
/// </summary>
internal sealed class FakeKrakenHandler : HttpMessageHandler
{
    private readonly List<(string PathContains, string Json)> _routes = [];

    public List<CapturedRequest> Calls { get; } = [];

    public void SetResponse(string pathContains, string json) =>
        _routes.Add((pathContains, json));

    protected override async Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request, CancellationToken ct)
    {
        var path = request.RequestUri!.PathAndQuery;
        var body = request.Content is not null
            ? await request.Content.ReadAsStringAsync(ct)
            : string.Empty;

        var hasApiSign = request.Headers.TryGetValues("API-Sign", out _);
        Calls.Add(new CapturedRequest(path, body, hasApiSign));

        foreach (var (contains, json) in _routes)
        {
            if (path.Contains(contains, StringComparison.OrdinalIgnoreCase))
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(json, Encoding.UTF8, "application/json")
                };
        }

        return new HttpResponseMessage(HttpStatusCode.NotFound)
        {
            Content = new StringContent($"Geen route geconfigureerd voor: {path}")
        };
    }
}

internal record CapturedRequest(string Path, string Body, bool HasApiSign);
