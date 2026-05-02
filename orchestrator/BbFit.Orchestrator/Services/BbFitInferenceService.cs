using BbFit.Orchestrator.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace BbFit.Orchestrator.Services;

public sealed class BbFitInferenceService : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _seqLen;
    private readonly int _inputSize;

    public BbFitInferenceService(IConfiguration config, ILogger<BbFitInferenceService> logger)
    {
        var modelPath = config["BbFit:ModelPath"] ?? "bbfit.onnx";
        _seqLen     = config.GetValue<int>("BbFit:SequenceLength", 64);
        _inputSize  = config.GetValue<int>("BbFit:InputSize", 33);

        _session   = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();

        logger.LogInformation("ONNX model geladen: {Path}, input={Name} [{Seq}x{Feat}]",
            modelPath, _inputName, _seqLen, _inputSize);
    }

    // features: flat float[seqLen * inputSize], volgorde [t0_f0..t0_f32, t1_f0..t1_f32, ...]
    public TradingSignal Predict(float[] features)
    {
        var tensor = new DenseTensor<float>(features, [1, _seqLen, _inputSize]);

        using var results = _session.Run(
            [NamedOnnxValue.CreateFromTensor(_inputName, tensor)]);

        var actionLogits = results[0].AsEnumerable<float>().ToArray();
        var sideLogits   = results[1].AsEnumerable<float>().ToArray();
        var equityDelta  = results[2].AsEnumerable<float>().First();

        var action     = (TradeAction)ArgMax(actionLogits);
        var side       = (TradeSide)ArgMax(sideLogits);
        var confidence = Softmax(actionLogits)[ArgMax(actionLogits)];

        return new TradingSignal(action, side, equityDelta, confidence, DateTimeOffset.UtcNow);
    }

    private static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }

    private static float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        var exp   = logits.Select(x => MathF.Exp(x - max)).ToArray();
        float sum = exp.Sum();
        return exp.Select(x => x / sum).ToArray();
    }

    public void Dispose() => _session.Dispose();
}
