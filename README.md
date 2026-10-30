# PiPER データ変換・Fine-tuning・Checkpoint共有 操作マニュアル

## 1. 概要と操作の流れ

本リポジトリは、PiPERの実機教示データを使うための次の機能を提供します。

1. 単腕・双腕のMCAPをLeRobot v2.1形式へ変換する。
2. π0／π0.5をFull／LoRAでFine-tuningする（全8構成）。
3. 学習済みCheckpointを設定・正規化統計とともにパッケージ化し、推論PCへ転送する。

操作順序は **環境構築 → 外部資産の準備 → データ変換 → 学習設定 → 正規化統計 → 学習 → 包装 → 転送** です。既存のLeRobotデータを使用する場合、変換は不要です。

コマンドには実行場所を明記しています。**ホスト**はUbuntuの通常端末、**Docker内**は `sh run.sh` で入った端末です。例のデータ名・実験名は利用するデータに合わせて変更してください。

```text
project_minimal/
├── README.md
├── build.sh / run.sh
├── Dockerfile / compose.yaml / requirements.lock
├── configs/{single,bimanual}/       学習YAML・MCAP schema
├── conversion/                     単腕・双腕の変換とデータ検証
├── training/                       共通学習入口・設定・adapter
├── scripts/                        起動・変換・学習・包装・転送
├── openpi/                         同梱モデル・学習・推論コード
├── datasets/{mcap,lerobot}/         利用者が配置・生成するデータ
├── models/                         Baseモデル・tokenizer
├── assets/                         正規化統計と署名
├── checkpoints/                    学習Checkpoint
├── logs/                           学習設定・metadata・ログ
└── exports/                        転送用archiveとSHA256
```

データ・重み・統計・学習出力はGitに含まれません。同梱OpenPIを使用し、別バージョンへ置き換えないでください。出典・再配布時の表記は `SOURCE_PROVENANCE.md`、`THIRD_PARTY_NOTICES.md`、`openpi/LICENSE`、`openpi/LICENSE_GEMMA.txt` を参照してください。

## 2. 前提条件と環境構築

### 2.1 必要な環境

- Ubuntu等のLinuxホスト、Git、Docker Engine、Docker Compose plugin。
- JAX CUDA 12環境と対象GPUに対応するNVIDIA Driver。
- DockerへGPUを公開できるNVIDIA Container Toolkit。
- ホスト利用者のDockerアクセス権、ネットワーク接続、十分なVRAM・ディスク容量。
- GUIで別ウィンドウを開く場合のみTerminator。SSHやGUIなしでも利用できます。

OSパッケージやドライバの導入は各PCの管理手順に従ってください。Dockerイメージ内のPython依存は `requirements.lock` で固定しています。ホスト側に学習用Pythonライブラリを導入する必要はありません。

**確認した学習PCの環境（2026年10月30日）** は以下です。最低要件を示すものではありません。

| 項目 | 確認した環境 |
|---|---|
| OS | Ubuntu 24.04.5 LTS |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition ×1 |
| GPUメモリ | 97,887MiB |
| NVIDIA Driver | 595.99.02 |
| Docker Engine / Compose | 29.9.2 / v5.6.1 |
| コンテナPython | 3.11.13 |
| JAX / JAXLIB | 0.5.3 / 0.5.3 |
| Flax / Orbax Checkpoint | 0.10.2 / 0.11.13 |
| PyTorch / TensorStore | 2.7.1 / 0.1.74 |
| LeRobot | v2.1データ形式、依存commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |

LeRobotのdistribution版表示は0.1.0ですが、互換性の基準は上記commitとv2.1データ形式です。GPUの種類によって必要なドライバや利用可能な構成は異なります。

### 2.2 cloneとビルド

**ホスト：** リポジトリへのアクセス権を用意して実行します。

```bash
cd ~
git clone git@github.com:matu711/project_minimal.git
cd ~/project_minimal

docker version
docker compose version
docker ps
nvidia-smi

sh build.sh
```

`sh build.sh` は必要な保存ディレクトリを作り、`tuning:local` をビルドします。初回はPython・CUDA関連ライブラリの取得に時間と容量が必要です。Docker APIへの権限がない場合は、ホスト側のグループ・ログインセッションを確認してください。スクリプトは権限を自動変更しません。

### 2.3 Dockerへ接続する

**ホスト：** 次のいずれかを実行します。

```bash
cd ~/project_minimal
sh run.sh
```

```bash
# 現在のターミナルで接続する場合
cd ~/project_minimal
sh run.sh --here
```

GUIセッションがありTerminatorが導入済みなら、新しい「PiPER Fine-tuning」ウィンドウで接続します。SSH・GUIなし・Terminator未導入では現在の端末を使用します。Terminatorを利用する場合の導入例は次のとおりです。

**ホスト：**

```bash
sudo apt install terminator
```

専用コンテナ名は `project-minimal-tuning-tuning-1`、サービス名は `tuning` です。既存の専用コンテナは再利用し、停止中なら起動します。実行ユーザーはホストUID/GIDを引き継ぎ、通常の学習をrootで行いません。ユーザー名が未登録の場合のみ、コンテナ内の名前解決情報を追加します。

**Docker内：** 作業位置とユーザーを確認します。

```bash
cd /workspace/project_minimal
pwd
id
whoami
printf '%s\n' "$HOME"
```

HOMEは `/workspace/project_minimal/.cache/tuning` です。既存コンテナの `OPENPI_ROOT` が古い場所を示す場合は、学習前に現在の同梱ソースへ合わせます。

**Docker内：**

```bash
export OPENPI_ROOT=/workspace/project_minimal/openpi
```

### 2.4 GPU確認と終了・再接続

**Docker内：** GPU一覧に加えて、JAXのGPU演算が同期完了することを確認してください。

```bash
nvidia-smi
python3 - <<'PY'
import jax
import jax.numpy as jnp
print("JAX:", jax.__version__)
print("devices:", jax.devices())
assert jax.default_backend() == "gpu", "GPUが認識されていません"
x = jnp.ones((256, 256), dtype=jnp.float32)
y = (x @ x).block_until_ready()
print("result:", float(y[0, 0]), "devices:", y.devices())
PY
```

CPUで計算できただけではGPU確認になりません。ROCm・TPUの初期化に関するINFO表示だけで判断せず、CUDAデバイス・backend・同期演算の結果を確認します。

**Docker内：** Bashから抜けるには次を実行します。専用コンテナ本体は起動したままです。

```bash
exit
```

再接続はホストで再び `sh run.sh` または `sh run.sh --here` を実行します。

## 3. Baseモデルとtokenizerの準備

Base重み・tokenizer・教示データはcloneだけでは取得されません。使用するモデルだけを用意してください。

| 資産 | 取得元 | Docker内の配置先 |
|---|---|---|
| π0 Base | `gs://openpi-assets/checkpoints/pi0_base/params` | `models/openpi-assets/checkpoints/pi0_base/params/` |
| π0.5 Base | `gs://openpi-assets/checkpoints/pi05_base/params` | `models/openpi-assets/checkpoints/pi05_base/params/` |
| tokenizer | `gs://big_vision/paligemma_tokenizer.model` | `models/big_vision/paligemma_tokenizer.model` |

表の配置先は `/workspace/project_minimal` からの相対パスです。Baseは各約12GB、両方で約24.46GBの取得実績があります。Checkpoint保存には別の空き容量が必要です。モデル・tokenizerの利用条件も確認してください。

既存キャッシュがなければ、同梱の取得関数でダウンロードできます。公開GCSへの匿名取得を使用します。**以下は大容量の取得コマンドです。既に配置済みなら再実行は不要です。**

**Docker内：π0.5を使用する場合**

```bash
cd /workspace/project_minimal
python3 - <<'PY'
from openpi.shared.download import maybe_download
print(maybe_download("gs://openpi-assets/checkpoints/pi05_base/params", gs={"token": "anon"}))
print(maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
PY
```

**Docker内：π0を使用する場合**

```bash
cd /workspace/project_minimal
python3 - <<'PY'
from openpi.shared.download import maybe_download
print(maybe_download("gs://openpi-assets/checkpoints/pi0_base/params", gs={"token": "anon"}))
print(maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
PY
```

`OPENPI_DATA_HOME` はComposeで `/workspace/project_minimal/models` に設定されます。別の方法でコピーする場合も、`params/` のmanifest・metadata・内部データファイルをすべて保持してください。

学習YAMLの `base_params_path` に上表のローカルパスを指定すると、その重みを使用します。`null` の場合はモデルに応じた公式GCSが指定され、未キャッシュなら取得を試みます。tokenizerも未キャッシュなら取得を試みます。

## 4. MCAPからLeRobotへ変換する

### 4.1 入力と収録仕様

**ホスト：** 収録済みMCAPと対応metadataを次のように配置します。

```text
~/project_minimal/datasets/mcap/my_single/   単腕の収録データ
~/project_minimal/datasets/mcap/my_dual/     双腕の収録データ
```

変換対象はROS1の `sensor_msgs/JointState` と `sensor_msgs/CompressedImage` を含むMCAPです。topic、関節名、単位は `configs/single/conversion.json`／`configs/bimanual/conversion.json` と一致する必要があります。JointStateの名前はschemaに定義された7要素の順序を要求します。

| 項目 | 単腕 | 双腕 |
|---|---|---|
| State / Action | 7D / 7D | 14D / 14D |
| 並び | 関節6軸→Gripper | 左関節6軸→左Gripper→右関節6軸→右Gripper |
| 保存値 | 絶対関節角rad、Gripper m | 同左 |
| 画像key | `front`, `wrist` | `front`, `left_wrist`, `right_wrist` |
| 元カメラの対応 | front／wrist | front／wrist1＝左／wrist2＝右 |

MCAPでは各streamの共通時間範囲に名目時刻グリッドを作り、header時刻の最近傍を選択します。ロボット値はグリッドから30ms以内、画像は100ms以内、選択したState／Action間の時刻幅は30ms以内を要求します。双腕では左右を含むロボットstream全体の時刻幅を検査します。

schemaは20Hz recorder用です。出力FPSは記録metadataから解決し、ない場合は警告のうえ20FPSを使用します。入力間でFPSが異なる場合は停止します。**双腕学習は20FPSを要求するため、変換結果の `meta/info.json` を確認してください。** 欠損値を補間して正常扱いする処理は行いません。

### 4.2 単腕の変換

**Docker内：** 入力を配置した後に実行します。例では2カメラを640×480へ揃えます。

```bash
cd /workspace/project_minimal
sh scripts/convert.sh \
  --robot single \
  --dataset my_single \
  --input datasets/mcap/my_single \
  --output datasets/lerobot/local/my_single \
  --task 'Pick up the object and place it in the box.' \
  --resize 640 480
```

単腕経路は互換性維持のため、中間rawディレクトリも生成します。この例では `datasets/lerobot/local/my_single.raw/` です。異なるカメラ解像度を入力する場合はresizeを明示してください。

### 4.3 双腕の変換

**Docker内：**

```bash
cd /workspace/project_minimal
sh scripts/convert.sh \
  --robot bimanual \
  --dataset my_dual \
  --input datasets/mcap/my_dual \
  --output datasets/lerobot/local/my_dual \
  --task 'Use both arms to pick up the object and place it in the box.'
```

双腕ではカメラごとに異なる解像度を保持できます。サイズを指定する場合、`--resize WIDTH HEIGHT` は全カメラへ適用します。現在のwriterは `quality.frame_valid` をboolで付与しますが、全行trueの列があるだけでは収録時刻の品質を保証しません。

### 4.4 引数・Task・出力の確認

| 引数 | 内容 |
|---|---|
| `--robot single\|bimanual` | 必須。ロボット構成 |
| `--dataset NAME` | 必須。repo IDは `local/NAME`。NAMEは単一ディレクトリ名 |
| `--input PATH` | MCAPファイル／ディレクトリ。複数指定可能。省略時は `datasets/mcap/NAME` |
| `--output PATH` | 省略時は `datasets/lerobot/local/NAME` |
| `--task TEXT` | 言語指示の明示指定 |
| `--mcap-schema PATH` | 別の確認済みMCAP schemaを指定 |
| `--resize WIDTH HEIGHT` | 画像サイズを統一 |
| `--validate-only` | 入力検査のみ。完成LeRobotデータは出力しない |
| `--timing-log PATH` | 収録時刻診断の保存先指定（主にraw入力） |

Taskは **`--task` → episodeの `meta.json` → `dataset_meta.json`** の順で取得し、欠落・空文字はエラーです。dataset／episode metadataのFPSやTaskを持ち運び、検証のために書き換えないでください。

単腕MCAPでは `--validate-only` でも中間rawと時刻診断ログを生成することがあります。完成LeRobotの生成を省く指定であり、完全な読み取り専用操作ではありません。

MCAPと同期済みrawの混在入力はできません。raw入力経路は記録順序を保ち、名目timestampを出力します。150ms等の時刻警告が出た場合、出力timestampの間隔だけで品質を判断せず、時刻診断・元metadataを確認してください。

正常終了時のLeRobotは概ね次の構造です。

```text
local/NAME/
├── meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl}
├── data/chunk-*/episode_*.parquet
├── videos/chunk-*/observation.images.*/episode_*.mp4
└── provenance/                     変換・検証の記録
```

**Docker内：** 出力先を合わせてmetadataを確認します。

```bash
python3 -m json.tool datasets/lerobot/local/my_dual/meta/info.json
python3 -m json.tool datasets/lerobot/local/my_dual/provenance/conversion.json
```

既存出力への上書きは禁止です。失敗時は標準エラーの対象stream・行番号・画像名を確認してください。`.partial-*` は未完成出力なので学習へ渡さず、完成データへ名前変更しないでください。

## 5. 全8構成のFine-tuning

### 5.1 正式Configと既定値

YAMLの `model` は `pi0`／`pi05`、`method` は `full`／`lora`、`robot` は `single`／`bimanual` です。Config名は `<model>_piper_<robot>_<method>` となります。

| Robot | Model | 方式 | 正式Config | Config名 | Batch | 総step | Horizon |
|---|---|---|---|---|---:|---:|---:|
| Single | π0 | Full | `configs/single/pi0_full.yaml` | `pi0_piper_single_full` | 1 | 30000 | 20 |
| Single | π0 | LoRA | `configs/single/pi0_lora.yaml` | `pi0_piper_single_lora` | 1 | 30000 | 20 |
| Single | π0.5 | Full | `configs/single/pi05_full.yaml` | `pi05_piper_single_full` | 1 | 30000 | 20 |
| Single | π0.5 | LoRA | `configs/single/pi05_lora.yaml` | `pi05_piper_single_lora` | 1 | 30000 | 20 |
| Bimanual | π0 | Full | `configs/bimanual/pi0_full.yaml` | `pi0_piper_bimanual_full` | 8 | 40000 | 50 |
| Bimanual | π0 | LoRA | `configs/bimanual/pi0_lora.yaml` | `pi0_piper_bimanual_lora` | 8 | 40000 | 50 |
| Bimanual | π0.5 | Full | `configs/bimanual/pi05_full.yaml` | `pi05_piper_bimanual_full` | 8 | 40000 | 50 |
| Bimanual | π0.5 | LoRA | `configs/bimanual/pi05_lora.yaml` | `pi05_piper_bimanual_lora` | 8 | 40000 | 50 |

共通の既定値は、保存間隔1000step、ログ間隔20step、worker数2、AdamW、peak LR＝1e-5、warmup＝1000step、decay＝40000step、最終LR＝1e-6です。単腕の総stepが30000でも、LR decay設定は40000です。

単腕テンプレートのdatasetは `local/piper_real_pickup_50epi`、双腕は `local/dnp1_20261015` です。これらのデータは同梱されません。以下の例は変換例の `my_single`／`my_dual` を明示指定します。

### 5.2 実験用YAMLを用意する

**Docker内：** 本番実験ごとにコピーし、エディタで設定します。コピー先が既にある場合、`cp -n` は上書きしません。

```bash
cp -n configs/bimanual/pi05_lora.yaml configs/bimanual/my_pi05_lora.yaml
```

| キー | 設定する内容 |
|---|---|
| `data_collection.dataset_repo_id` | 例：`local/my_dual`。CLIの `--dataset my_dual` でも指定可能 |
| `exp_name` | 新しい単一ディレクトリ名。例：`my_dual_pi05_lora_run01` |
| `batch_size` | VRAMに合わせる。まず1で短い実行を確認 |
| `cuda_visible_devices` | 単腕既定は文字列 `"0"`、双腕は `null`。1GPUなら文字列 `"0"` を明示可能 |
| `num_train_steps` | 学習終了時の総step数 |
| `lr_schedule` | `peak_lr`、`warmup_steps`、`decay_steps`、`decay_lr` |
| `action_horizon` | 単腕20／双腕50が既定。変更するとモデル形状・統計署名に影響 |
| `save_interval` / `log_interval` | 保存・ログ間隔。最初の保存より前に中断すると再開用状態が残らない |
| `method` | `full` または `lora`。テンプレートと合わせる |
| `lora_rank` | `null` で言語16／Action expert32。正整数指定は両者へ適用 |
| `base_params_path` | 例：`models/openpi-assets/checkpoints/pi05_base/params` |
| `assets_dir` | 統計の保存元ディレクトリ。既定 `assets`。独立した実験では新しい場所を指定可能 |
| `output_dir` | Checkpointの保存元ディレクトリ。既定 `checkpoints` |
| `resume` / `overwrite` | 新規は `false` / `false`。共有入口は `overwrite: true` を許可しない |

相対パスはプロジェクトルート基準です。CLIのdataset指定は、その実行のYAMLを上書きする優先指定です。統計・検査・学習のすべてで同じdatasetを使用してください。

π0は通常のState入力・標準正規化、π0.5はdiscrete State入力・quantile正規化です。最大token長の既定はπ0が48、π0.5が200です。モデル内部はState／Actionとも32Dへpaddingします。

Fullは全パラメータを更新し、正式YAMLではEMA 0.99を使用します。LoRAは `gemma_2b_lora`／`gemma_300m_lora` を使用し、EMAは無効です。言語／Action expertのBaseはfreeze filterで凍結しますが、画像encoder等も更新対象で、adapterだけの学習ではありません。

単腕では関節6軸、双腕では左右の関節6軸ずつを観測Stateからの差分Actionへ変換し、Gripperは絶対値を維持します。双腕は20FPS・14D・3カメラ・verified provenanceを検査し、`quality.frame_valid` で学習anchorを選択します。他のカメラ品質フラグをすべて自動除外する仕様ではありません。

### 5.3 設定・正規化統計・実Batchを確認する

**Docker内：** 上で編集した実験用YAMLを使用します。

```bash
sh scripts/train.sh --config configs/bimanual/my_pi05_lora.yaml --check-config
sh scripts/train.sh --config configs/bimanual/my_pi05_lora.yaml --norm-stats-only
sh scripts/train.sh --config configs/bimanual/my_pi05_lora.yaml --validate-only
```

- `--check-config`：モデル・LoRA・保存先等の設定構築だけを確認。データやGPU学習の成功判定ではありません。
- `--norm-stats-only`：実データからState／Actionのmean・std・q01・q99と署名を生成。
- `--validate-only`：実データBatch、関節差分、Gripper、画像・padding・正規化の逆変換を確認。先に統計を作成してください。

統計は `<assets_dir>/<Config名>/local/<dataset>/norm_stats.json` と `tuning_norm_config.json` に保存します。署名にはdataset metadata、robot、model、horizon、batch size等が含まれます。既存統計は署名一致時に再利用します。学習開始時も、統計がなければ計算します。

単腕／双腕、π0／π0.5の統計を混用しないでください。dataset・batch等を変更した新規実験では、新しい `assets_dir` を用意してください。`--recompute-norm-stats` は既存統計を更新する操作なので、既存学習・Resume用の統計へ安易に使用しないでください。Resume中の再計算は禁止されています。

### 5.4 全8構成の学習コマンド

**Docker内：** 正式Configを直接指定する8通りです。使用前に各YAMLの実験名・VRAMに関係する設定を確認してください。別実験用にコピーした場合は `--config` をそのファイルへ置き換えます。**8構成を同時に起動せず、必要な1構成を選択してください。**

```bash
# Single / π0 / Full
sh scripts/train.sh --config configs/single/pi0_full.yaml --dataset my_single

# Single / π0 / LoRA
sh scripts/train.sh --config configs/single/pi0_lora.yaml --dataset my_single

# Single / π0.5 / Full
sh scripts/train.sh --config configs/single/pi05_full.yaml --dataset my_single

# Single / π0.5 / LoRA
sh scripts/train.sh --config configs/single/pi05_lora.yaml --dataset my_single

# Bimanual / π0 / Full
sh scripts/train.sh --config configs/bimanual/pi0_full.yaml --dataset my_dual

# Bimanual / π0 / LoRA
sh scripts/train.sh --config configs/bimanual/pi0_lora.yaml --dataset my_dual

# Bimanual / π0.5 / Full
sh scripts/train.sh --config configs/bimanual/pi05_full.yaml --dataset my_dual

# Bimanual / π0.5 / LoRA
sh scripts/train.sh --config configs/bimanual/pi05_lora.yaml --dataset my_dual
```

実験用YAMLの場合の通常起動と、console・GPU CSVも保存する前景起動は次のとおりです。**どちらか一方を選びます。**

**Docker内：**

```bash
sh scripts/train.sh --config configs/bimanual/my_pi05_lora.yaml
```

```bash
python3 scripts/foreground_train.py --config configs/bimanual/my_pi05_lora.yaml
```

前景起動ラッパーには `--config` を渡します。dataset等はYAMLに設定してください。学習ログは `logs/<Config名>/<exp_name>/train.log`、実効設定は `training.yaml`、包装用の参照情報は `metadata.json` に保存します。ラッパーでは追加で日時付きconsoleログ・GPU CSVを保存します。

### 5.5 VRAM・ディスクと検証範囲

整理後ソースで8構成すべてについて、**実データ・事前学習重みを使用したGPU学習2step、保存、別プロセスResume追加1step** の成功実績があります。条件はBatch 1、単腕horizon20／双腕50です。正式YAMLの双腕Batch 8を検証済みとするものではありません。

同GPUでLoRAはGPU全体約36,000MiB、Fullは約93,600MiBを観測しました。Fullでは `XLA_CLIENT_MEM_FRACTION=0.93` を使用しました。汎用の最低VRAM要件や他GPUでの成功保証ではありません。

**Docker内：Fullを同条件で実行する場合の例。YAMLのBatchは別途1に設定します。**

```bash
cp -n configs/bimanual/pi05_full.yaml configs/bimanual/my_pi05_full.yaml
# エディタでdataset・新しいexp_name・Batch 1等を設定してから起動する
XLA_CLIENT_MEM_FRACTION=0.93 sh scripts/train.sh --config configs/bimanual/my_pi05_full.yaml
```

上の `my_pi05_full.yaml` は、実行前に `configs/bimanual/pi05_full.yaml` からコピーして設定するファイルです。LoRA用YAMLをFull成功として扱わないでください。

CheckpointはLoRAで約9～10GB、Fullで約43～45GBの保存実績があります。複数の保持step・包装・展開用にも容量が必要です。自動保持は最新に加えて既定5000step周期を残す設計なので、1個分だけの空き容量では不十分です。

**ホスト：** 実行前にGPU利用状況と保存領域を確認します。

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
df -h ~/project_minimal
```

## 6. 学習の中断とResume

中断は学習端末でCtrl+Cを使用します。保存は非同期なので、通常完了時は `Waiting for checkpoint manager to finish` の処理が終了してから端末を閉じてください。保存前の更新は復元できません。

再開には次を保持します。

- 同じ実験の、確定済みCheckpoint（`params/`、`train_state/`、`_CHECKPOINT_METADATA`等）。
- `logs/<Config名>/<exp_name>/training.yaml` とmetadata。
- 元の正規化統計と一致する署名。

ディレクトリ番号はループ添字です。40,000step完了時は通常 `39999/`、内部の `train_state.step` は40000です。`resume` は最新の保存状態から再開し、CLIに特定stepの選択引数はありません。

**Docker内：** 例の実験名を実際のものへ変更し、実効設定からResume用YAMLを作成します。

```bash
cp -n \
  logs/pi05_piper_bimanual_lora/my_dual_pi05_lora_run01/training.yaml \
  configs/bimanual/my_pi05_lora_resume.yaml

python3 - <<'PY'
from pathlib import Path
import yaml
path = Path("configs/bimanual/my_pi05_lora_resume.yaml")
params = yaml.safe_load(path.read_text())
params["resume"] = True
# 完了済み学習を延長する場合だけ、総step数を現在のStepより大きくする。
# params["num_train_steps"] = 45000
path.write_text(yaml.safe_dump(params, sort_keys=False))
PY

sh scripts/train.sh --config configs/bimanual/my_pi05_lora_resume.yaml
```

元のdataset・model・method・rank・horizon・batch・保存先・実験名を変えないでください。変更が許可されるのは `resume`、`num_train_steps`、`no_wandb` です。総step数は「追加step数」ではなく、終了時の総数です。

**保存状態がない、またはstepディレクトリが `0` だけの場合、現行実装は重みのResumeを行いません。** `resume: true` だけで復元成功と判断せず、復元ログとStepの連続性を確認してください。Checkpointがない中断実験は記録を保持し、新しい実験名でBaseから始めます。既存metadataを削除して上書きしないでください。

## 7. Checkpointをパッケージ化する

### 7.1 Config・実験・stepの指定

包装は全8構成共通です。`--config-name` はYAMLのファイル名ではなく、5.1の **Config名** を指定します。`--experiment` は実際の `exp_name`、`--step` は保存ディレクトリの番号です。stepを省略すると最新の確定済み数値ディレクトリを選択します。

学習YAMLの `output_dir` が既定以外でも、`logs/.../metadata.json` を使って実際の保存先を解決します。ログmetadataと正規化統計を消さないでください。

**ホスト：** SSH設定を使う後続の転送と揃えるため、包装もホストで行う例です。Python 3の標準ライブラリだけで実行できます。

```bash
cd ~/project_minimal
config_name=pi05_piper_bimanual_lora
experiment=my_dual_pi05_lora_run01
step=39999

sh scripts/package_checkpoints.sh \
  --config-name "$config_name" --experiment "$experiment" --step "$step" --dry-run

# 表示された入力・出力先を確認し、未作成の場合に生成する
sh scripts/package_checkpoints.sh \
  --config-name "$config_name" --experiment "$experiment" --step "$step"
```

別構成ではConfig名と実験名・保存stepを変更します。実際に保存されたstepだけを指定してください。Docker内で包装する場合は `/workspace/project_minimal` で同じコマンドを使用できます。

今回の40,000step学習済み双腕π0.5 LoRAについて、対象選択だけを確認する例です。

**ホスト：**

```bash
sh scripts/package_checkpoints.sh \
  --config-name pi05_piper_bimanual_lora \
  --experiment dnpdemo1_20261028_pi05_lora_restart_20261028_213936 \
  --step 39999 --dry-run
```

この例のarchiveは既に存在するため、再生成は不要です。既存archiveまたはSHA256ファイルがある場合、包装処理は上書きせずエラーで停止します。

### 7.2 保存内容と確認方法

出力は `exports/<Config名>-<exp_name>-step<番号>.tar.gz` と、同名に `.sha256` を付けたファイルです。パッケージ形式は次のとおりです。

```text
<archive内>/
├── checkpoint/<step>/
│   ├── _CHECKPOINT_METADATA
│   ├── params/                         推論用の全パラメータ木
│   ├── train_state/                    Step・Optimizer等の再開状態
│   └── assets/local/<dataset>/norm_stats.json
├── metadata.json
├── training.yaml
├── norm_stats.json
├── tuning_norm_config.json             存在する場合
├── bimanual_norm_config.json           存在する場合
└── MANIFEST.sha256.json                 全収録ファイルのSHA256
```

LoRAはBaseを含む全パラメータ木を保存する形式で、adapter-onlyではありません。LoRAマージや重みの形式変換は行いません。Fullの推論用paramsは学習時のEMAが有効ならEMAを使用します。Optimizer等も同梱するため、推論専用より大きなパッケージです。推論専用の軽量化オプションはありません。

包装処理は、収録前の各ファイルSHA256とarchive内の各ファイルを照合し、最後にarchive全体のSHA256を出力します。

**ホスト：** 7.1と同じ変数を設定した端末で、生成物を確認します。

```bash
package="${config_name}-${experiment}-step${step}.tar.gz"
ls -lh "exports/$package" "exports/$package.sha256"
(cd exports && sha256sum -c "$package.sha256")
tar -tzf "exports/$package"
```

## 8. 推論PCへscp転送する

### 8.1 接続と保存先

現在の転送先は `dnp218@163.221.133.164`、保存先は `/home/dnp218/piper_checkpoints/` です。ホスト側でSSH認証を設定してください。秘密鍵をDockerへコピーする必要はありません。

**学習PCのホスト：** 転送先ディレクトリを作成します。

```bash
ssh dnp218@163.221.133.164 'mkdir -p /home/dnp218/piper_checkpoints'
```

認証エラーの場合は、管理者とSSH鍵・アカウントの設定を確認してください。転送完了・ハッシュ一致を確認する前に転送元を削除しないでください。

### 8.2 今回のstep39999を転送する例

**学習PCのホスト：**

```bash
cd ~/project_minimal
archive=exports/pi05_piper_bimanual_lora-dnpdemo1_20261028_pi05_lora_restart_20261028_213936-step39999.tar.gz
sha_file="${archive}.sha256"

(cd exports && sha256sum -c "$(basename "$sha_file")")
scp -- "$archive" "$sha_file" \
  dnp218@163.221.133.164:/home/dnp218/piper_checkpoints/
```

### 8.3 別モデルにも使える汎用例

**学習PCのホスト：** 対象Config・実験名・stepだけを変更します。archiveは先に7章の手順で作成してください。

```bash
cd ~/project_minimal
config_name=pi0_piper_single_full
experiment=my_single_pi0_full_run01
step=29999
package="${config_name}-${experiment}-step${step}.tar.gz"

(cd exports && sha256sum -c "$package.sha256")
scp -- "exports/$package" "exports/$package.sha256" \
  dnp218@163.221.133.164:/home/dnp218/piper_checkpoints/
```

既存の転送スクリプトを使う場合も、ホストで実行します。次は同じ変数を使い、archive SHA256を検査してからscpします。

**学習PCのホスト：**

```bash
sh scripts/transfer_checkpoints.sh \
  --config-name "$config_name" --experiment "$experiment" --step "$step" \
  --transport scp \
  --destination dnp218@163.221.133.164:/home/dnp218/piper_checkpoints/
```

### 8.4 推論PCで整合性確認・展開する

**学習PCのホスト：** 推論PCへログインします。

```bash
ssh dnp218@163.221.133.164
```

**推論PCのホスト：** 以下は今回のstep39999の例です。別構成ではarchive名と新しい展開ディレクトリ名を変更します。

```bash
cd /home/dnp218/piper_checkpoints
package=pi05_piper_bimanual_lora-dnpdemo1_20261028_pi05_lora_restart_20261028_213936-step39999.tar.gz

ls -lh "$package" "$package.sha256"
sha256sum -c "$package.sha256"
tar -tzf "$package"

# 未使用のディレクトリ名を指定する。既に存在する場合は展開を停止する
mkdir dnpdemo1_pi05_lora_step39999 && \
  tar -xzf "$package" -C dnpdemo1_pi05_lora_step39999
```

`sha256sum` が `OK` であることを確認してから展開してください。必要なら、展開後の全収録ファイルもmanifestと照合します。

**推論PCのホスト：**

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
root = Path("/home/dnp218/piper_checkpoints/dnpdemo1_pi05_lora_step39999")
manifest = json.loads((root / "MANIFEST.sha256.json").read_text())
for name, expected in manifest.items():
    digest = hashlib.sha256()
    with (root / name).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"SHA256 mismatch: {name}")
print(f"Verified {len(manifest)} files")
PY
```

推論PCでは `/home/dnp218/piper_vla_runtime_share/` の新Checkpoint共通ローダー（`checkpoint_package.py`）を使用します。単腕・双腕、π0・π0.5、Full・LoRAを学習設定と照合し、パッケージ内の統計を使ってください。推論側には互換ソース・ライブラリ・tokenizerが別途必要です。これらと実機制御・カメラ取得はCheckpoint archiveに含まれません。

今回の双腕π0.5 LoRA step39999は、学習PCの同梱OpenPIによるGPU復元・推論を確認済みで、推論PCの共通ローダーでの成功報告もあります。ただし、推論PCの実装を直接参照した照合と、残る7構成の推論PC上の実行は未確認です。本READMEは推論側の起動CLIやROS制御を定義しません。

## 9. 保存先とホスト／Dockerの対応

プロジェクトルート全体をbind mountします。ホストで置いたファイルはDockerから同じ相対位置で参照できます。

| 用途 | ホスト側（既定） | Docker内（既定） |
|---|---|---|
| プロジェクト | `~/project_minimal` | `/workspace/project_minimal` |
| MCAP | `~/project_minimal/datasets/mcap/NAME` | `/workspace/project_minimal/datasets/mcap/NAME` |
| LeRobot | `~/project_minimal/datasets/lerobot/local/NAME` | `/workspace/project_minimal/datasets/lerobot/local/NAME` |
| 学習Config | `~/project_minimal/configs/<robot>/*.yaml` | `/workspace/project_minimal/configs/<robot>/*.yaml` |
| Baseモデル等 | `~/project_minimal/models/` | `/workspace/project_minimal/models/` |
| 正規化統計 | `~/project_minimal/assets/<Config名>/local/NAME/` | `/workspace/project_minimal/assets/<Config名>/local/NAME/` |
| Checkpoint | `~/project_minimal/checkpoints/<Config名>/<exp_name>/<step>/` | `/workspace/project_minimal/checkpoints/<Config名>/<exp_name>/<step>/` |
| 学習ログ・実効設定 | `~/project_minimal/logs/<Config名>/<exp_name>/` | `/workspace/project_minimal/logs/<Config名>/<exp_name>/` |
| archive | `~/project_minimal/exports/` | `/workspace/project_minimal/exports/` |

`assets_dir`・`output_dir` 等を変更した場合は、その設定が優先されます。ホストとDockerで別々のCheckpointを持つわけではありません。包装スクリプトはmetadata内の `/workspace/project_minimal` をホストのプロジェクト位置へ対応付けます。推論PCへの転送後は、元PCの絶対パスがそのまま利用できるとは限りません。

## 10. トラブルシューティング

| 症状 | 確認・対応 |
|---|---|
| Dockerへ接続できない | ホストでDocker daemon、socketとグループ、現在のログインセッションを確認。権限緩和や `sudo docker` を回避する |
| Terminatorが閉じる／ウィンドウが見つからない | `sh run.sh --here` で接続経路を分け、Docker・GUI・起動元環境を確認 |
| GPU backendがCPU | ホスト／Docker両方の `nvidia-smi`、Toolkit、Driver、`cuda_visible_devices` を確認 |
| GPU OOM | 他プロセスのVRAM、Batch、horizon、Full／LoRAを確認。必要なら新規設定でBatch 1から検証 |
| MCAPのtopic・関節名エラー | 対応する `configs/<robot>/conversion.json` と実収録仕様を照合 |
| 同期許容値・収録時刻の警告 | エラーに示すstream・行を確認。metadata書換えで隠さず、収録品質を評価 |
| 画像shapeエラー | 元解像度を確認。必要なら `--resize` で統一 |
| 双腕のFPS・provenanceエラー | `meta/info.json`、`provenance/conversion.json`、`quality.frame_valid` を確認 |
| dataset・Base重みがない | dataset repo IDと配置先、YAMLの `base_params_path` を確認 |
| 正規化設定が変わった | dataset・metadata・Batch・horizon等の署名差分を確認。新実験では統計の新規保存先を用意 |
| `Experiment metadata already exists` | 同じ実験名の記録あり。保存Checkpointがあれば正しいResume、なければ新しい実験名で開始 |
| Resume設定が一致しない | 元の `logs/.../training.yaml` を基準にし、許可項目以外を変更しない |
| 包装対象がない | Config名・exp_name・保存step・ログmetadata・確定済みCheckpointを確認 |
| archiveが既にある | 上書きは禁止。既存archiveとSHA256を検査し、必要なら別実験／stepを包装 |
| scpが認証エラー | ホスト側のSSH設定・アカウント・接続先権限を確認 |
| SHA256不一致 | 利用・展開を止め、転送元の検証と再転送を行う |

**ホスト：Dockerアクセスの確認**

```bash
id
id -nG
getent group docker
ls -l /var/run/docker.sock
docker ps
docker compose ps
```

**Docker内：保存先とログの確認（例の実験名は実際のものへ変更）**

```bash
ls -lah checkpoints/pi05_piper_bimanual_lora/my_dual_pi05_lora_run01
tail -n 60 logs/pi05_piper_bimanual_lora/my_dual_pi05_lora_run01/train.log
python3 -m json.tool logs/pi05_piper_bimanual_lora/my_dual_pi05_lora_run01/metadata.json
```

**Docker内：正式CLIの引数を確認**

```bash
sh scripts/convert.sh --help
sh scripts/train.sh --help
sh scripts/package_checkpoints.sh --help
sh scripts/transfer_checkpoints.sh --help
```

データセット・Checkpoint・Base重み・ログ・exportsは、整理やエラー対処のために削除しないでください。特にログの `training.yaml`・`metadata.json` と統計署名は、Resume・包装に必要な実行資産です。
