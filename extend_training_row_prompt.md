# Taak: Uitbreid WriteTrainingRow() met TAengine interne indicatoren

## Context

Dit project traint een LSTM om het gedrag van een Bollinger Band trading bot na te bootsen.
De bot (C# / Crypteum) logt per candle een JSON-rij via `WriteTrainingRow()` in
`TradeActionsLookAheadFree.cs`. Die rijen worden samengevoegd tot `lstm_merged.csv` en
gebruikt als traindata voor de LSTM.

**Het probleem:** `WriteTrainingRow()` logt de ruwe OHLC-data en de trade-beslissing
(`actionTaken`), maar NIET de technische indicatoren die de bot intern berekent via
`TAengine` (Bollinger Bands, RSI, SMA, etc.). De LSTM mist daardoor de exacte features
die de bot zelf ziet op het moment van beslissen.

## Doel

Breid `WriteTrainingRow()` uit zodat het de interne TAengine-waarden op `signalIndex`
meellogt. Het resultaat wordt straks door een Python pipeline omgezet naar LSTM-features.

---

## Relevante bestanden

### `TradeActionsLookAheadFree.cs` — de `WriteTrainingRow()` functie (vereenvoudigd):

```csharp
void WriteTrainingRow(
    OhlcDto signalBar,
    OhlcDto executionBar,
    decimal currentTimestamp,
    double executionPriceValue,
    double tradeActionRaw,
    double lastTradeValue,
    int actionTaken,
    string runId)
{
    // ... berekent inPosition, positionValue, netEquity, tradeSide ...

    var row = new
    {
        runId,
        timestamp = currentTimestamp,
        interval, intervalMinutes, observedIntervalMinutes,
        signalTimestamp, executionTimestamp,
        signalOpen, signalHigh, signalLow, signalClose,
        executionOpen, executionHigh, executionLow, executionClose,
        executionPrice, tradeActionRaw, tradeSide, lastTrade, actionTaken,
        tradingCapital, assetsHeld, inPosition, entryPrice, positionValue, netEquity,
        buyCount, sellCount, wins, losses,
        totalTradedNotional, feePerSide, cost
    };

    Log<TrainingLogMarker>.Write.Info(JsonConvert.SerializeObject(row, Formatting.None));
}
```

De functie wordt aangeroepen in de loop als:

```csharp
var deltaAssets = bbTecAn.GetDeltaAssets();   // al beschikbaar
var lastTrades  = bbTecAn.GetLastTrades();    // al beschikbaar
var tradeAction = deltaAssets[signalIndex];
var lastTrade   = lastTrades[signalIndex];

WriteTrainingRow(signalBar, currentBar, currentTimestamp,
    executionPrice, tradeAction, lastTrade, actionTaken, runId);
```

`bbTecAn` is een instantie van `TAengine` — op het moment van de aanroep zijn alle
arrays al gevuld t/m `signalIndex`.

### `TAengine.cs` — beschikbare public getters:

```csharp
public double[] GetTimeSerie()       // close prices
public double[] GetDeltaValues()     // delta close
public double[] GetSma()             // Simple Moving Average (main period)
public double[] GetEma()             // Exponential Moving Average
public double[] GetDeviation()       // Standard deviation (sd_)
public double[] GetUpperBand()       // Bollinger upper band
public double[] GetLowerBand()       // Bollinger lower band
public double[] GetBandWidth()       // BB bandwidth
public double[] GetBandWidthDelta()  // BB bandwidth delta
public double[] GetRsi()             // RSI
public double[] GetStochRsi()        // Stochastic RSI
public double[] GetLastTrades()      // lastTrade signal (1=buy, 2=sell)
public double[] GetDeltaAssets()     // trade action signal
```

---

## Wat je moet doen

### Stap 1: Breid `WriteTrainingRow()` uit

Voeg de volgende parameters toe aan de functie-signatuur:

```csharp
void WriteTrainingRow(
    OhlcDto signalBar,
    OhlcDto executionBar,
    decimal currentTimestamp,
    double executionPriceValue,
    double tradeActionRaw,
    double lastTradeValue,
    int actionTaken,
    string runId,
    // nieuw:
    double sma,
    double ema,
    double upperBand,
    double lowerBand,
    double deviation,
    double bandWidth,
    double bandWidthDelta,
    double rsi,
    double stochRsi
)
```

Voeg de nieuwe velden toe aan het anonieme `row` object:

```csharp
sma, ema, upperBand, lowerBand, deviation,
bandWidth, bandWidthDelta, rsi, stochRsi
```

### Stap 2: Lees de waarden op `signalIndex` uit bbTecAn

Op de aanroeplocatie in de loop, lees de waarden op `signalIndex`:

```csharp
WriteTrainingRow(signalBar, currentBar, currentTimestamp,
    executionPrice, tradeAction, lastTrade, actionTaken, runId,
    sma:           bbTecAn.GetSma()[signalIndex],
    ema:           bbTecAn.GetEma()[signalIndex],
    upperBand:     bbTecAn.GetUpperBand()[signalIndex],
    lowerBand:     bbTecAn.GetLowerBand()[signalIndex],
    deviation:     bbTecAn.GetDeviation()[signalIndex],
    bandWidth:     bbTecAn.GetBandWidth()[signalIndex],
    bandWidthDelta:bbTecAn.GetBandWidthDelta()[signalIndex],
    rsi:           bbTecAn.GetRsi()[signalIndex],
    stochRsi:      bbTecAn.GetStochRsi()[signalIndex]
);
```

### Stap 3: Controleer array bounds

Zorg dat `signalIndex` geldig is voor alle arrays (check op `signalIndex >= 0` en
`signalIndex < array.Length`) voordat je de waarden opvraagt. Bij een ongeldige index:
gebruik `0.0` als default.

---

## Output die ik verwacht

Geef terug:

1. **De volledige gewijzigde `WriteTrainingRow()` functie** in C#
2. **De volledige gewijzigde aanroep** in de loop (inclusief de bounds-check)
3. **Een lijst van de nieuwe JSON-velden** die in de output verschijnen, zodat ik de
   Python pipeline (`add_indicators.py` of een nieuw script) kan aanpassen om deze
   velden direct te gebruiken als features in plaats van ze te herberekenen

## Wat ik er mee doe

De output van jou gebruik ik als volgt:
- De C# code vervangt de huidige `WriteTrainingRow()` in `TradeActionsLookAheadFree.cs`
- De lijst van nieuwe JSON-velden gebruik ik om in Python de feature-kolommen te
  definiëren voor de LSTM-trainpipeline
- `add_indicators.py` kan dan worden vereenvoudigd of verwijderd omdat de indicatoren
  nu rechtstreeks uit de bot komen i.p.v. herberekend te worden

## Belangrijk
- Wijzig de bestaande velden niet — voeg alleen toe
- Zorg dat de JSON-keys exact overeenkomen met de Python verwachting (snake_case)
- De aanroep in de loop moet ook de `WriteTrainingRow()` aanroepen op de plek waar
  `inPosition=0` en geen trade plaatsvindt (de "hold" rijen) — ook daar moeten de
  indicatorwaarden worden meegegeven
