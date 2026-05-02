# BbFit Orchestrator - Context voor nieuwe sessie

## Doel
Bouw een .NET 10 ASP.NET Core orchestrator die:
1. Real-time marktdata ontvangt via Kraken WebSocket v2
2. Die data invoert in het bb-fit LSTM model (via ONNX inference)
3. Op basis van het signaal orders plaatst en monitort via Kraken REST API
4. State synchroniseert naar een fallback instantie via SignalR
5. Draaibaar is als Docker container op een NVIDIA DGX Spark (ARM64, Ubuntu 24.04)
6. Later deploybaar op Amazon ECS

## Repository
- GitHub: dwight-modiwirijo/bb-fit
- Werkbranch: feature/orchestrator
- Lokaal op DGX Spark: /home/dwyte/bb-fit/

## bb-fit model (ONNX)
- Model: /home/dwyte/bb-fit/bbfit.onnx
- Metadata: /home/dwyte/bb-fit/bbfit.meta.json
- input_size=33, hidden_size=512, num_layers=3, sequence_length=64
- action_classes=3: Hold=0, Buy=1, Sell=2
- trade_side_classes=3: None=0, Long=1, Short=2
- ONNX output[0]=action_logits, output[1]=trade_side_logits, output[2]=net_equity_delta

## .NET project locatie
/home/dwyte/bb-fit/orchestrator/
- BbFit.slnx
- BbFit.Orchestrator/
  - BbFit.Orchestrator.csproj  (packages: OnnxRuntime 1.25.1, Websocket.Client 5.3.0)
  - Program.cs                  KLAAR
  - appsettings.json            NOG AANPASSEN
  - Models/TradingSignal.cs     KLAAR
  - Services/BbFitInferenceService.cs  KLAAR
  - Services/KrakenWebSocketClient.cs  NOG TE DOEN
  - Services/OrderManager.cs           NOG TE DOEN
  - Hubs/StateHub.cs                   NOG TE DOEN

## Dotnet PATH instellen na elke sessie
export DOTNET_ROOT=$HOME/.dotnet
export PATH=$PATH:$HOME/.dotnet

## Nog te doen

### KrakenWebSocketClient.cs
- BackgroundService
- Verbind wss://ws.kraken.com/v2
- Subscribe ticker/OHLC (handelspaar via appsettings)
- Bouw float[64*33] feature vector
- Roep BbFitInferenceService.Predict() aan
- Geef signaal door aan OrderManager

### OrderManager.cs
- Kraken REST API v2
- PlaceOrderAsync(), CancelOrderAsync(), GetOpenOrdersAsync()
- HMAC-SHA512 authenticatie (ApiKey + ApiSecret uit appsettings/env)
- Max positiegrootte + stop-loss

### StateHub.cs
- SignalR hub op /state
- Broadcast open orders, laatste signaal, actieve positie naar fallback

### appsettings.json
{
  "BbFit": { "ModelPath": "bbfit.onnx", "SequenceLength": 64, "InputSize": 33 },
  "Kraken": { "ApiKey": "", "ApiSecret": "", "TradingPair": "BTC/EUR", "MaxPositionSize": 0.01 }
}

### Dockerfile
- Multi-stage: mcr.microsoft.com/dotnet/sdk:10.0 -> mcr.microsoft.com/dotnet/aspnet:10.0
- ARM64 (DGX Spark)
- Kopieer bbfit.onnx + bbfit.meta.json in container
- Expose port 8080

## Kraken API
- WebSocket v2: wss://ws.kraken.com/v2
- REST API v2: https://api.kraken.com/0/
- Auth: HMAC-SHA512 over nonce + encoded payload

## Features
33 features per tijdstap uit Kraken marktdata + technische indicatoren.
Zie normalization_stats.json (feature_columns) en add_indicators.py.
