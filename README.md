# PiPER：LeRobot変換・Fine-tuning・Checkpoint共有

単腕／双腕PiPERのMCAP→LeRobot変換、OpenPI π0／π0.5のFull／LoRA学習、Checkpointの転送用パッケージ化を提供します。シミュレータ、Mock学習、FAST、DROID、旧実験プリセットは対象外です。

## 構成

```text
build.sh / run.sh / Dockerfile / compose.yaml / requirements.lock
configs/{single,bimanual}/     8学習YAMLとMCAP schema
conversion/                   単腕・双腕と共通処理
training/                     設定構築・検査・学習入口
scripts/                      Docker、変換、学習、包装・転送
openpi/                       同梱モデル・学習・推論・clientとライセンス
tests/                        小規模な設定・実writer・Orbax回帰テスト
```

ソースは同梱OpenPI snapshotを使用します。上流の別バージョンで置き換えないでください。モデル構造・依存の正本は本リポジトリと `requirements.lock` です。出典は `SOURCE_PROVENANCE.md`、第三者表記は `THIRD_PARTY_NOTICES.md` と `openpi/LICENSE*` を参照してください。

## 環境構築と起動

Ubuntu、Docker EngineとCompose plugin、ホストのNVIDIAドライバ、NVIDIA Container Toolkitが必要です。Python 3.11、JAX/JAXLIB 0.5.3、Flax 0.10.2、Orbax 0.11.13等はDocker buildで固定依存として導入します。ビルドにはネットワークが必要です。

```sh
git clone git@github.com:matu711/project_minimal.git
cd project_minimal
sh build.sh
sh run.sh             # GUI＋Terminatorなら新しいウィンドウ
sh run.sh --here      # 現在のターミナル。SSHでも利用可能
```

Dockerアクセス権はホスト側で準備してください。スクリプトはsudo dockerやソケットの権限緩和を行いません。Terminatorは任意で、未導入なら `sudo apt install terminator` と案内して現在の端末へ接続します。

専用サービスは `tuning:local` / `project-minimal-tuning-tuning-1`、コンテナ内は `/workspace/project_minimal`。ホストUID/GIDで実行し、ユーザー名未登録時だけコンテナ内のアカウント情報を追加します。HOMEは `.cache/tuning`。既存の専用コンテナは再利用し、他コンテナは操作しません。

以下のPython・学習・変換コマンドは、原則コンテナ内のプロジェクトルートで実行します。同梱ソースを優先するので、別ワークスペースは不要です。既存コンテナに古いOPENPI_ROOTを設定している場合は `export OPENPI_ROOT=/workspace/project_minimal/openpi` としてください。

## 外部資産の準備

データ・重み・統計・学習出力はGitに含まれません。

| 資産 | 取得元／配置 |
|---|---|
| 単腕・双腕MCAP | 利用者が用意。`datasets/mcap/<収録名>/` |
| LeRobot | 本変換で生成、または互換データを `datasets/lerobot/local/<dataset>/` へ配置 |
| π0 Base | `gs://openpi-assets/checkpoints/pi0_base/params` → `models/openpi-assets/checkpoints/pi0_base/params/` |
| π0.5 Base | `gs://openpi-assets/checkpoints/pi05_base/params` → `models/openpi-assets/checkpoints/pi05_base/params/` |
| PaliGemma tokenizer | `gs://big_vision/paligemma_tokenizer.model` → `models/big_vision/paligemma_tokenizer.model` |

Base重みは各約12GB、両方で約24.46GBの取得実績があります。公開GCSから匿名で取得できた実績がありますが、ネットワーク・配布条件を確認してください。GCS対応ツールでディレクトリ全体を再帰コピーし、Orbaxのmanifestや内部ファイルを省略しないでください。モデル・tokenizerの利用条件は各ライセンスに従ってください。

YAMLの `base_params_path` に上表のローカルパスを指定すると、追加ダウンロードなしで読み込めます。`null` の場合は公式GCSが指定され、未キャッシュなら取得を試みます。tokenizerも未キャッシュなら取得を試みます。`OPENPI_DATA_HOME` は既定でプロジェクトの `models` を使用します。

## MCAPからLeRobotへ

```sh
sh scripts/convert.sh --robot single --dataset my_single \
  --input datasets/mcap/my_single --task 'Pick up the object.'
sh scripts/convert.sh --robot bimanual --dataset my_dual \
  --input datasets/mcap/my_dual --task 'Pick up the object with both arms.'
```

既定出力は `datasets/lerobot/local/<dataset>`。既存出力は上書きしません。schemaは `configs/<robot>/conversion.json`。異なるtopic名は確認済みschemaを別ファイルで用意し `--mcap-schema PATH` で指定してください。

単腕は7D・2カメラ、双腕は左7D→右7D・3カメラ。関節rad／Gripper mの絶対値を保存します。20Hz recorderのschemaを検査し、記録metadataから名目FPSを決定、同期・データ検証・provenance保存を行います。双腕学習は20FPSを要求します。実収録間隔に欠落がある場合、名目timestampが正常でも時間品質が良好とは限りません。時刻診断を確認してください。

Taskは `--task` → episode metadata → dataset metadata の順で取得し、欠落時はエラーです。`--input` は複数指定可能。`--validate-only`、`--resize WIDTH HEIGHT`、`--timing-log PATH` に対応します。単腕で異なるカメラ解像度を揃える場合もresizeを明示してください。raw入力も互換性のため維持しています。

## 全8構成のFine-tuning

```text
configs/single/pi0_full.yaml       configs/single/pi0_lora.yaml
configs/single/pi05_full.yaml      configs/single/pi05_lora.yaml
configs/bimanual/pi0_full.yaml     configs/bimanual/pi0_lora.yaml
configs/bimanual/pi05_full.yaml    configs/bimanual/pi05_lora.yaml
```

テンプレートを新しいYAMLへコピーし、`data_collection.dataset_repo_id`、`exp_name`、`base_params_path`、batch size、step数と保存間隔を設定してください。例示datasetは別途必要です。PC固有の実験YAMLはGitへ追加しないでください。既存実験名は再使用せず、初回は `resume: false`、再開時だけ `true` にします。

```sh
sh scripts/train.sh --config configs/bimanual/my_run.yaml --check-config
sh scripts/train.sh --config configs/bimanual/my_run.yaml --norm-stats-only
sh scripts/train.sh --config configs/bimanual/my_run.yaml --validate-only
sh scripts/train.sh --config configs/bimanual/my_run.yaml
# consoleとGPU CSVも保存する前景実行
python3 scripts/foreground_train.py --config configs/bimanual/my_run.yaml
```

`--norm-stats-only` は実データから統計と署名を `assets/<config>/local/<dataset>/` へ生成します。異なるrobot・model・データ変換の統計は混用しないでください。Resume時は同じ統計と署名が必要です。

単腕horizon 20、双腕50、モデル内部32Dへpadding。関節6軸ずつを観測からの差分へ変換し、Gripperは絶対値を維持します。双腕は `quality.frame_valid` をanchor選択に使用し、verified provenanceを要求します。カメラ品質の全フラグを自動除外する仕様ではありません。

π0.5はdiscrete State入力。LoRAはlanguage rank 16、action expert rank 32が既定（`lora_rank: null`）。freeze filterに従いBaseの言語／Action expertを凍結しますが、画像encoder等は更新対象です。Fullは全パラメータを更新しEMAを保存します。

RTX PRO 6000 Blackwell Max-Q約96GBで、Batch 1のLoRAはGPU全体約36GB、Fullは約94GBの使用実績があります。Fullは検証時 `XLA_CLIENT_MEM_FRACTION=0.93` が必要でした。これは機種固有の実績であり、デフォルトBatch 8の動作保証ではありません。Full用には十分なVRAM・ディスクを確保し、他のGPU学習と同時実行しないでください。

Checkpointは `checkpoints/<config>/<exp_name>/<番号>/`、学習設定とmetadataは `logs/<config>/<exp_name>/`。双方を保持してください。ディレクトリ番号はループ添字で、serialized state.stepより1小さい仕様です。保存前の中断からは更新を復元できません。Resume用YAMLは同じ実験・設定を維持し、`resume` と終了step数等の許可項目だけを変更してください。

## Checkpointパッケージ

```sh
sh scripts/package_checkpoints.sh --config-name pi05_piper_bimanual_lora \
  --experiment my_experiment --step 39999 --dry-run
# 確認後、--dry-runを外して生成
sh scripts/package_checkpoints.sh --config-name pi05_piper_bimanual_lora \
  --experiment my_experiment --step 39999
```

従来の形式を維持します。

```text
checkpoint/<step>/{params/,train_state/,assets/,_CHECKPOINT_METADATA}
metadata.json
training.yaml
norm_stats.json
tuning_norm_config.json / bimanual_norm_config.json（存在するもの）
MANIFEST.sha256.json
```

`exports/<config>-<experiment>-step<step>.tar.gz` と外部 `.sha256` を生成し、全収録ファイルのSHA256を照合します。Baseを含む学習後パラメータ木、LoRA、Optimizer、統計を保存します。重みの変換・マージ・変更は行いません。推論専用の除外オプションはありません。

推論PCには同じソースと依存環境、tokenizerを別途用意してください。データセットと元Base配布重みは推論に不要です。`openpi.policies.policy_config.create_trained_policy(config, '<展開先>/checkpoint/<step>')` が学習時と同じモデル・LoRA構成でparamsとassets内統計を読みます。YAML中の学習PCの絶対パスは移植先に合わせて構築し直してください。自動書換えは行いません。単腕・双腕のadapterを取り違えないでください。

双腕の通常Policy入力は `state` 14D、RGB uint8のCHW画像 `images.cam_high`、`images.cam_left_wrist`、`images.cam_right_wrist` と `prompt`。出力は `[50,14]` の絶対目標（関節は差分から復元、Gripperは絶対値）です。実機制御・カメラ取得は別システムです。

任意の外部転送は `sh scripts/transfer_checkpoints.sh --experiment NAME --step STEP --destination user@host:/path`。転送前にarchive SHA256を再検証します。公開範囲や転送先を確認して実行してください。

## 回帰確認

`tests/` はMock GPU成功判定を含めず、8構成の実設定、少量fixtureの実LeRobot writer・正規化、実Orbax保存復元を確認します。検証環境へpytest 8.3.4を用意し、コンテナ内で `JAX_PLATFORMS=cpu python3 -m pytest tests -q`。この結果だけで実GPU学習成功とは判断しません。

## Gitへ含めないもの

`datasets/`、`models/`、`assets/`、`checkpoints/`、`logs/`、`exports/`、`validation/`、`.cache/`、bytecode、認証情報は配布ソースから除外します。これらはローカルでは保持します。特にログのmetadata・training YAMLは包装とResumeに必要です。
