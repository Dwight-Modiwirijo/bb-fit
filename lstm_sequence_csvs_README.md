# LSTM sequence CSV handoff

The actual merged source file `lstm_merged.csv` is not present in this chat session, so the three CSVs here are schema-only placeholders.

Use this command on the DGX or locally once `lstm_merged.csv` is available:

```bash
python build_lstm_sequence_csvs.py --input /path/to/lstm_merged.csv --output-dir /path/to/output --sequence-length 64
```

Outputs:
- `lstm_train_sequences.csv`
- `lstm_validation_sequences.csv`
- `lstm_test_sequences.csv`
- `lstm_sequence_build_summary.json`

The builder follows the agreed policy:
- group by `runId`
- sort by `timestamp`
- split in time order
- no windows crossing split boundaries
- no global shuffle
