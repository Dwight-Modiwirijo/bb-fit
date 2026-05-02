using BbFit.Orchestrator.Models;
using BbFit.Orchestrator.Services;
using Microsoft.AspNetCore.SignalR;

namespace BbFit.Orchestrator.Hubs;

public sealed class StateHub : Hub
{
    private readonly OrderManager _orders;

    public StateHub(OrderManager orders)
    {
        _orders = orders;
    }

    // Stuur volledige toestand bij nieuwe verbinding
    public override async Task OnConnectedAsync()
    {
        var snap = _orders.Portfolio.Snapshot(0m);
        await Clients.Caller.SendAsync("Portfolio", snap);
        await base.OnConnectedAsync();
    }

    // Fallback instantie roept dit aan om synchronisatie te bevestigen
    public async Task Ping() =>
        await Clients.Caller.SendAsync("Pong", DateTimeOffset.UtcNow);
}
