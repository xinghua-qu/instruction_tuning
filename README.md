# ZEBRA: A Machine Learning Framework for Efficient Algorithm Implementation

## Config
The config supports reading from both `yaml` file and `argparse`.

```
python3 train.py --config ./config/instruction_tuning_bloom_560m.yaml -v --trainer.lr 0.00002
```

This will enable the user definition `--trainer.lr 0.00002` to override the content in `./config/instruction_tuning_bloom_560m.yaml` .