# How to be *Stoic* in your own way

Framework and data for training *Stoic*.

## 1. Environment setup

Choose one option, then install training dependencies.

### `venv`

```bash
python -m venv .stoic-env
source .stoic-env/bin/activate
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

Download URL: https://zenodo.org/records/19100654

Use the helper script:

```bash
chmod +x stoic_train/download_zenodo_data.sh
./stoic_train/download_zenodo_data.sh /path/to/data_root_parent
```

The script downloads `data_root.zip` and unpacks it into the directory you pass in.

Example:

```bash
./stoic_train/download_zenodo_data.sh ./data
```

If you download the archive manually, unzip it like this:

```bash
cd stoic
unzip -o stoic_train/data/data_root.zip -d stoic_train/data
unzip -o stoic_train/data/data_file.csv.zip -d stoic_train/data
```

After this, your dataset is expected at:

- root directory: `stoic_train/data/data_root/`
- metadata file: `stoic_train/data/data_root/data_file.csv`

## 3. Optional: create a train/val split from existing train rows

If your CSV currently has only `train` + `benchmark` (or similar), you can split
only the current `train` rows into `train`/`val` based on IDs in a `cluster_label` column:

```bash
python -m stoic_train.make_train_val_split \
  --data-file stoic_train/data/data_root/data_file.csv \
  --val-ratio 0.1 \
  --seed 993
```

Notes:
- `--split-feature` defaults to `cluster_label`.
- Set `--val-ratio 0` to keep all current train rows as `train`.
- Rows with mixed IDs from both groups are marked as `unassigned`.

## 4. Configure data + training config

Use the config in `configs/config_train_stoic.yaml` and set:

- `data.init_args.root` -> graph dataset root
- `data.init_args.data_file` -> CSV with split/labels metadata
- logger/checkpoint paths if needed

The training entrypoint is `python -m stoic_train.train`, powered by Lightning CLI.

## 5. Run training

### Single run from config

```bash
python -m stoic_train.train fit --config stoic_train/configs/config_train_stoic.yaml
```

### Resume from checkpoint

```bash
python -m stoic_train.train fit \
  --config stoic_train/configs/config_train_stoic.yaml \
  --ckpt_path path/to/checkpoint.ckpt
```

### Override selected settings from CLI

```bash
python -m stoic_train.train fit \
  --config stoic_train/configs/config_train_stoic.yaml \
  --data.init_args.batch_size=32 \
  --trainer.max_epochs=50
```

## 6. Notes

- Most configs are set to `wandb` offline mode by default and must be synchronized manually; adjust logger settings if you want online tracking.
