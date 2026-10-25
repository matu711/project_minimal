# PiPER共有用チューニング環境

実機PiPERの教示データ変換、OpenPIのπ0／π0.5学習、チェックポイントの出力・転送を行います。基本操作は次の2コマンドです。

```sh
cd ~/project_minimal
sh build.sh     # 初回、またはDocker／依存関係／同梱OpenPIの更新時
sh run.sh       # コンテナの起動・再利用 → Terminatorで操作
```

`run.sh` はホストのTerminatorを開き、その中でコンテナの `sh` に接続します。Terminator未導入・GUIなしの場合は現在の端末で接続します。`sh run.sh --attach` で現在の端末へ直接接続することもできます。

Terminatorはホストにインストールしてください（Ubuntu: `sudo apt install terminator`）。起動オプションは[Terminator公式ドキュメント](https://gnome-terminator.readthedocs.io/en/latest/advancedusage.html)に従っています。

`exit` は接続したシェルだけを終了します。コンテナの主プロセスは `sleep infinity` なので稼働を続けます。`run.sh` は再ビルドやイメージの取得を行いません。イメージがない場合は `sh build.sh` を案内して終了します。停止中のコンテナは同じコンテナを起動し、起動済みなら再利用します。ビルド後に稼働中コンテナを新イメージへ切り替える場合は、学習が終わってから明示的に再作成してください。

どちらもPOSIX sh対応で、別ディレクトリから `sh /path/to/project_minimal/run.sh` と実行できます。サービスは `tuning`、Composeプロジェクトは `project-minimal-tuning`、イメージは `tuning:local`、通常のコンテナ名は `project-minimal-tuning-tuning-1` です。既存の `piper-pi0-sim:111` や既存コンテナは操作しません。別コピーを同時利用する場合は、この専用プロジェクト名も分ける必要があります。

## 必要な環境

- Linux x86_64、NVIDIA GPU、対応するホストドライバ、Docker EngineとComposeプラグイン、NVIDIA Container Toolkit。
- ビルド時のインターネット接続と十分なディスク空き容量。重み・データはイメージに含みません。
- GPUライブラリは固定済みJAX 0.5.3／PyTorch 2.7.1のCUDA 12系wheelから導入します。Pythonは3.11.13です。
- VRAMの目安は既存コードの記載でLoRA約22.5GB超、Full約70GB超です。画像数、horizon、batch sizeに依存します。複数GPU利用時はYAMLの `fsdp_devices` とbatch sizeも調整してください。設定例の値は動作・収束の保証ではありません。

GPU予約は[Docker公式のCompose GPU設定](https://docs.docker.com/compose/how-tos/gpu-support/)に従っています。ホストのドライバやDocker設定のインストール・変更は、このプロジェクトのスクリプトでは行いません。

## 構成と保存先

```text
project_minimal/
├── build.sh / run.sh / Dockerfile / compose.yaml
├── requirements.lock             # Python依存の固定、LeRobotはコミット固定
├── conversion/                   # 共通pipeline・reader・同期・writer＋各robotのadapter
├── training/                     # 共通入口、設定構築、adapter、双腕データ検証
├── configs/{single,bimanual}/     # 各pi0／pi05 × lora／full
├── scripts/                      # 変換・学習・圧縮・転送
├── datasets/{mcap,lerobot}/       # 元データと変換結果
├── checkpoints/                  # Orbaxチェックポイント
├── exports/                      # 圧縮ファイルとSHA256
├── logs/                         # 有効な学習YAML・メタデータ・ログ
├── models/                       # ダウンロードしたbase modelのキャッシュ
├── assets/                       # データセットごとの正規化統計
├── tests/
└── openpi/                       # 必要な同梱OpenPIソース、ライセンスを保持
```

コンテナ内の作業場所は `/workspace/project_minimal`。プロジェクト全体を同じ場所にbind mountし、データ・チェックポイント・ログ・キャッシュをホストへ永続化します。`build.sh`／`run.sh` はホストUID/GIDを渡すため、生成物もホストユーザーで所有します。

Dockerのビルドコンテキストは `.dockerignore` の許可リストで限定しています。旧 `piper_pi0/` 一式、archive、シミュレータDocker設定、実機制御スクリプトは新イメージに入りません。必要なPiPER処理は `conversion/` と `training/` に移植済みです。MuJoCo、gym-aloha、robosuite、dm-control、pymunkも導入しません。OpenPIの学習用adapter・transformは必要なので保持します。既存ソースの旧READMEは調査資料です。**起動にはこのREADMEを使用してください。**

OpenPIの必要なソースは `/opt/project_minimal/openpi` に取り込みます。別PCに `/workspace/openpi` や `/workspace/piper_pi0` を用意する必要はありません。`conversion/`・`training/`・YAMLはbind mount側にあるので、編集後すぐに使えます。同梱OpenPIの変更には再ビルドと専用コンテナの再作成が必要です。初期コピーの `piper_pi0/` は参照資料として保持していますが、現在の変換・学習処理には不要です。新しい配置は [conversion/README.md](conversion/README.md)、コピー元とライセンスの記録は [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) を参照してください。

## 最小コード一式を別PCへ共有する

```sh
sh scripts/package_project.sh --dry-run
sh scripts/package_project.sh
```

`exports/project_minimal-source.tar.gz` とSHA256を生成します。変換・学習コード、設定、Docker定義、必要なOpenPIソース、ライセンス、テストだけを含みます。旧 `piper_pi0/` 一式、sim、データ、重み、ログ、キャッシュ、認証ファイルは含めません。展開後はそのディレクトリで `sh build.sh`、`sh run.sh` を実行してください。既存archiveがある場合は `--output exports/別の名前.tar.gz` を指定します。archive内の `SHARED_SOURCE_SHA256.json` と外部の `.sha256` で整合性を確認できます。

## データ配置と変換

コンテナ内で実行します。既定入力は `datasets/mcap/<dataset>`、出力は `datasets/lerobot/local/<dataset>` です。

```sh
sh scripts/convert.sh --robot single --dataset dataset_name
sh scripts/convert.sh --robot bimanual --dataset dataset_name

# MCAP単体。記録taskがない場合は --task が必要
sh scripts/convert.sh --robot single --dataset single_mcap \
  --input /path/to/record.mcap --task 'Pick up the object.'
sh scripts/convert.sh --robot bimanual --dataset dual_mcap \
  --input /path/to/record.mcap --task 'Assembling a box.'

# 両腕ともRGB変換後にPillow BILINEARで指定サイズへ変更
sh scripts/convert.sh --robot single --dataset single_resized \
  --input /path/to/raw_dataset --resize 224 224
sh scripts/convert.sh --robot bimanual --dataset dual_resized \
  --input /path/to/raw_episode --resize 224 224

# datasetを作らず入力検証（rawの時刻ログは新規作成）
sh scripts/convert.sh --robot bimanual --dataset check --input /path/to/raw --validate-only
```

rawは収録時に対応付けられた行として扱い、不規則なフレーム間隔でも全記録行を順に保存します。実測時刻・時刻差・不規則区間を警告し、`<output>.raw_timing-UUID.json` と完成データ内 `provenance/raw_timing.json` に記録します。出力timestampは名目 `frame_index / fps` です。実測時計がなければ監査不能と警告します。欠損行の生成、ゼロ埋め、補間、MCAP復元、暗黙のepisode分割、画像の再選択は行いません。State/Actionの欠損・次元不正・NaN/Inf、画像欠損・破損はエラーです。

単腕は従来の `dataset_meta.json`、episodeの `meta.json`・`data.npz`、`front/000000.jpg`・`wrist/000000.jpg` 等を使用します。親にdataset metadataがあるepisode単体も指定できます。V1/V5/V6 EE pose、複数入力、`--action-mode`、`--include-failed`、`--config`、`--repo-id` を維持します。rawのCSV/capture時刻は変換順や画像対応を変えるためには使用しません。`20260825_224317` は450行のまま変換します。

双腕rawはepisodeの `meta.json`・`data.npz`・`frames.csv`・`front/`・`wrist1/`・`wrist2/` を使用し、CSVに記録された各行の画像を保持します。記録の右7→左7を出力の左7→右7へ並べ替え、カメラ名はfront／left_wrist／right_wristです。不規則raw `20261015_165042` も記録済み1,411行を変換し、欠損した2行は追加しません。画像/配列の実数の不一致や記録画像の欠損は拒否します。

FPSはdataset metadataの `int(fps)` を優先し、なければepisode metadata、どちらもなければ20と警告します。採用FPSが複数episode/入力間で異なる場合はエラーです。dataset FPSがある場合、episodeの実測FPSで上書きしません。LeRobot metadata・動画FPS・名目timestampは採用FPSで一致します。既存双腕学習の入力検査は20FPSを要求するため、別FPSのLeRobot出力をそのまま既存学習設定へ渡すことはできません。学習コードは変更していません。

両腕のtaskは **`--task` → episode metadataのtask → dataset metadataのtask → エラー** です。文字列の前後空白を含めそのまま保持します。`--input-task`、`--task-replace`、`tsku` alias、YAMLのtask fallback、汎用の既定文言は廃止しました。`--config` の出力先/repo推論は維持します。複数入力に別taskを付ける場合は記録metadataを持つ入力を使用するか、変換を分けてください。

MCAPはROS header時刻を使用し、ロボットと全カメラの共通範囲へ採用FPSのグリッドを置き、各ストリームで最近傍を独立に選びます。同距離は過去側、robot各30ms・camera100ms以内、選ばれたロボット時刻の最大差30ms以内を要求します。双腕4系統で条件を満たす別の組を探索する処理は廃止しました。許容差超過はエラーです。トピック・関節名・単位は `configs/<robot>/conversion.json`、別schemaは `--mcap-schema PATH` で指定します。schemaの20Hzは既存recorderの識別条件で、出力グリッドFPSは記録metadataから解決します。

単腕MCAPは従来のCRC・画像検査・SHA256確認とJPEG quality95の中間rawを維持し、`<output>.raw/` を新規作成します。`--validate-only` でもMCAPはこの抽出を行います。双腕は中間rawを作らずwriterへ渡します。画像resize未指定時は元サイズを保持し、指定時は全カメラを指定幅・高さへ変更します（アスペクト比の補正なし）。

既定の `joint_absolute` は絶対関節Action・絶対Gripperを保存します。単腕 `--action-mode ee_delta_axis_angle` はEE poseから差分を計算し末尾を1行落とす従来機能です。これは関節Actionとは別の表現です。学習時の関節delta化・絶対Gripper、双腕の互換列 `quality.frame_valid=true` は維持します。このvalid列は受け入れた記録行を示し、raw実時計の30ms/100ms再同期を保証する列ではありません。

共通入口は新規 `.partial-*` へ書き、基本検証成功後に正式名へrenameします。上書きは禁止し、不完全な出力を完成扱いしません。直接 `python -m conversion.single` の明示overwriteは従来どおりです。`--timing-log PATH` は新規ログ先を指定します。

構成は [conversion/README.md](conversion/README.md)、今回の差分と検証は [CONVERSION_ALIGNMENT.md](CONVERSION_ALIGNMENT.md) を参照してください。過去の復元・簡素化報告は変更履歴です。

## 学習と設定の確認

```sh
sh scripts/train.sh --robot single --model pi0  --method lora --check-config
sh scripts/train.sh --robot single --model pi0  --method full --check-config
sh scripts/train.sh --robot single --model pi05 --method lora --check-config
sh scripts/train.sh --robot single --model pi05 --method full --check-config
sh scripts/train.sh --robot bimanual --model pi0  --method lora --check-config
sh scripts/train.sh --robot bimanual --model pi0  --method full --check-config
sh scripts/train.sh --robot bimanual --model pi05 --method lora --check-config
sh scripts/train.sh --robot bimanual --model pi05 --method full --check-config
```

`--check-config` を外すとデータ検証・必要な正規化統計の計算・学習を実行します。`--dataset NAME` で `local/NAME` を選択できます。`--config configs/bimanual/pi05_lora.yaml` でも指定できます。CLIのrobot/model/methodはYAMLより優先するため、異なる条件では `exp_name` とbase modelなども確認してください。

```sh
sh scripts/train.sh --config configs/bimanual/pi05_lora.yaml --norm-stats-only
sh scripts/train.sh --config configs/bimanual/pi05_lora.yaml --validate-only
sh scripts/train.sh --config configs/bimanual/pi05_lora.yaml
nvidia-smi
```

`--validate-only` は1 batchのshape・有限値・paddingを検証し、学習しません。統計がない場合は正規化前の検証です。`--check-config` の成功は実データ・GPU・学習成功を意味しません。

YAMLでは `robot`、`model`、`method`、`data_collection.dataset_repo_id`、`data_dir`、`exp_name`、`base_params_path`、`batch_size`、`num_train_steps`、`lr_schedule`、`action_horizon`、`lora_rank`、`save_interval`、`output_dir`、`resume` を編集できます。パスはproject_minimal基準とし、共有用の学習パスはプロジェクト内に限定します。base modelのURIも指定可能です。既定は各モデルに対応する `gs://openpi-assets/checkpoints/{pi0,pi05}_base/params` で、必要に応じてダウンロードします。

LoRAは `gemma_2b_lora`／`gemma_300m_lora` と実際のfreeze filterを設定します。Fullは非LoRAモデルと全パラメータの学習設定を使います。`lora_rank: null` は既存variantのLanguage=16／Action Expert=32、整数指定は両方のrankを上書きします。単腕は7次元・2カメラ、双腕は14次元・3カメラで検証し、モデル入力は32次元にpadします。関節Actionは観測stateを基準にdelta化し、gripperは絶対値を維持します。双腕は無効な画像の開始点を除外し、将来Actionの20Hzグリッドを維持します。

単腕の初期horizon=20／batch=1は既存単腕設定、双腕のhorizon=50／batch=8は既存双腕設定を参考にしています。単腕は `cuda_visible_devices: "0"` で1GPUを使用します。複数GPUを使う場合は `"0,1"` などに変更し、batch sizeをGPU数の倍数にしてください。双腕の既定は `null`（コンテナから見える全GPU）。GPU数は `fsdp_devices` の倍数である必要があります。Fullや双腕π0の設定は実データによる収束確認が必要です。

## 保存とResume

チェックポイントは `checkpoints/<config名>/<exp_name>/<step>/`、統計は `assets/<config名>/local/<dataset>/`、有効な設定とログは `logs/<config名>/<exp_name>/` に保存します。ログには `training.yaml`、`metadata.json`、`train.log` が含まれます。上書きを防ぐため共有用trainerは `overwrite: true` を拒否します。

再開時は同じYAMLで `resume: true` を指定します。同じ実験名、モデル、保存先、統計、データセットを保持してください。学習ステップの上限は変更可能です。resumeで統計を再計算することはできません。新規実験で統計の再計算が必要なら `--norm-stats-only --recompute-norm-stats` を使います。

## チェックポイントの圧縮・検証・転送

```sh
sh scripts/package_checkpoints.sh --experiment bimanual_pi05_lora_example --dry-run
sh scripts/package_checkpoints.sh --experiment bimanual_pi05_lora_example
sh scripts/transfer_checkpoints.sh --experiment bimanual_pi05_lora_example \
  --destination user@host:/path/to/destination --dry-run
sh scripts/transfer_checkpoints.sh --experiment bimanual_pi05_lora_example \
  --destination user@host:/path/to/destination --transport rsync
```

確定済みの最新stepを選びます。`--step 1000` で指定でき、同名実験が複数あれば `--config-name pi05_piper_bimanual_lora` で限定します。圧縮にはparams・optimizerを含む選択step全体、学習YAML、正規化統計、モデル情報、各ファイルのSHA256 manifestを含め、実際のアーカイブ内の内容も再検証します。LoRAもOpenPIの全パラメータツリー形式で保存し、adapter単体と偽って扱いません。base modelのURI／パスはmetadataに明示します。base modelそのものは同梱しません。

転送前には `.tar.gz.sha256` を再検証します。転送先でも `sha256sum -c ARCHIVE.tar.gz.sha256` で確認してください。転送スクリプトは事前に作ったarchiveを使い、再圧縮しません。SSH鍵をイメージに入れたり自動マウントしたりしません。必要ならホストで転送スクリプトを実行し、ホスト側のSSH設定を使ってください。パッケージの自動展開・別PCでの自動復元は行いません。復元時はYAMLの保存先や統計の検証記録を移動先に合わせ、同じコード・依存関係を用意してください。

## エラー時の確認

- Docker接続エラー: ホストで `docker info` と利用権限を確認。
- GPU予約エラー: NVIDIA Container Toolkitとホストの `nvidia-smi` を確認。
- imageがない: `sh build.sh`。buildエラーは表示して非ゼロ終了します。
- GPUメモリ不足: batch size・horizonを見直し、Fullでは十分なVRAM／FSDPを用意。
- datasetがない: `datasets/lerobot/local/<name>/meta/info.json` とYAMLのIDを確認。`.partial-*` は学習に使えません。
- 同期エラー／未知のtopic: 元MCAPと収録metadataを確認。既存データは編集せず、別出力先で再検証。
- 統計の不一致: 新規実験用に明示的に再計算。resumeでは元の統計を復元。
- 変更したDockerイメージが反映されない: 学習終了後、専用サービスのみ `docker compose -p project-minimal-tuning up -d --no-build --force-recreate tuning` を実行。その後 `sh run.sh`。

検証結果と未検証事項は `VALIDATION.md` を参照してください。既存のコピー元を示す `SHARE_MANIFEST.json` は初期コピー時点の記録であり、今回の追加ファイルのmanifestではありません。

## テスト

コンテナ内で `sh scripts/test.sh -q` を実行します。pytest等のテスト用ライブラリは `/tmp` に配置し、Python環境本体は変更しません。テストはCPU上の小規模データ・モデルで、変換、統計、保存再開などを確認します。実モデルの学習完了を検証するものではありません。
