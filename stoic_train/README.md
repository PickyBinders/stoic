# How to be *Stoic* in your own way

Framework and data for training for *Stoic*.

## 1. Environment setup

Choose one option, then install training dependencies.

### `venv`

```bash
python -m venv .stoic-venv
source .stoic-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[training]"
```

### `conda` / `mamba`

```bash
mamba create -n stoic-train python=3.10 -y
mamba activate stoic-train
python -m pip install --upgrade pip
python -m pip install -e ".[training]"
```

## 2. Download training data

[Placeholder]

## 3. Configure data + training config

Use the config in `configs/config_train_stoic.yaml` and set:

- `data.init_args.root` -> graph dataset root
- `data.init_args.data_file` -> CSV with split/labels metadata
- logger/checkpoint paths if needed

The training entrypoint is `python -m stoic_train.train`, powered by Lightning CLI.

## 4. Run training

### Single run from config

```bash
python -m stoic_train.train fit --config configs/config_train_stoic.yaml
```

### Resume from checkpoint

```bash
python -m stoic_train.train fit \
  --config configs/config_train_stoic.yaml \
  --ckpt_path path/to/checkpoint.ckpt
```

### Override selected settings from CLI

```bash
python -m stoic_train.train fit \
  --config configs/config_train_stoic.yaml \
  --data.init_args.batch_size=32 \
  --trainer.max_epochs=50
```

## 5. Notes

- Most configs are set to `wandb` offline mode by default; adjust logger settings if you want online tracking.
