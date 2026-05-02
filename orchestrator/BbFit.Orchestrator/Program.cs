using BbFit.Orchestrator.Hubs;
using BbFit.Orchestrator.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSignalR();
builder.Services.AddSingleton<BbFitInferenceService>();
builder.Services.AddSingleton<OrderManager>();
builder.Services.AddHostedService<KrakenWebSocketClient>();

var app = builder.Build();

app.MapHub<StateHub>("/state");
app.MapGet("/health", () => Results.Ok(new { status = "healthy", time = DateTime.UtcNow }));

app.Run();
