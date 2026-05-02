#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


class MultiHeadLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, action_classes, trade_side_classes):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=lstm_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_size, action_classes)
        self.trade_side_head = nn.Linear(hidden_size, trade_side_classes)
        self.net_equity_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = self.dropout(output[:, -1, :])
        return {"action_logits": self.action_head(last_hidden), "trade_side_logits": self.trade_side_head(last_hidden), "net_equity_delta": self.net_equity_head(last_hidden).squeeze(-1)}


class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["action_logits"], out["trade_side_logits"], out["net_equity_delta"]


def infer_hyperparams(state_dict):
    ih = state_dict["lstm.weight_ih_l0"]
    hidden_size = ih.shape[0] // 4
    input_size = ih.shape[1]
    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1
    return {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers, "action_classes": state_dict["action_head.weight"].shape[0], "trade_side_classes": state_dict["trade_side_head.weight"].shape[0]}


def main():
    parser = argparse.ArgumentParser(description="Export bb-fit .pt checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalization-stats", default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = payload["model_state_dict"]
    params = infer_hyperparams(state_dict)

    print(f"input_size={params['input_size']} hidden_size={params['hidden_size']} num_layers={params['num_layers']} action_classes={params['action_classes']} trade_side_classes={params['trade_side_classes']}")

    model = MultiHeadLstm(**{k: params[k] for k in params}, dropout=args.dropout)
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, args.sequence_length, params["input_size"])
    torch.onnx.export(wrapper, dummy, args.output, input_names=["input"], output_names=["action_logits", "trade_side_logits", "net_equity_delta"], dynamic_axes={"input": {0: "batch_size"}, "action_logits": {0: "batch_size"}, "trade_side_logits": {0: "batch_size"}, "net_equity_delta": {0: "batch_size"}}, opset_version=17, do_constant_folding=True)

    meta = {**params, "sequence_length": args.sequence_length, "outputs": {"0": "action_logits", "1": "trade_side_logits", "2": "net_equity_delta"}}
    if args.normalization_stats:
        with open(args.normalization_stats) as f:
            meta["n_features"] = json.load(f).get("n_features")

    meta_path = Path(args.output).with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done: {args.output}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
