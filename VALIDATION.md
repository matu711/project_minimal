# 現在の変換仕様について

以下は共通化前の検証履歴です。現在の単腕仕様と検証は [SINGLE_COMPATIBILITY_RESTORE.md](SINGLE_COMPATIBILITY_RESTORE.md)、双腕簡素化の履歴は [CONVERSION_SIMPLIFICATION.md](CONVERSION_SIMPLIFICATION.md) を参照してください。旧recover／flag等の記載は現行機能を表しません。

# 検証結果と対応状況（2026-10-25）

今回の構成と操作方法は [README.md](README.md)、変換コードの役割は [conversion/README.md](conversion/README.md)、コピー元は [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) を参照してください。

## 完成した構成

```text
project_minimal/
├── build.sh / run.sh / Dockerfile / compose.yaml
├── requirements.lock
├── conversion/
│   ├── convert_mcap_to_lerobot.py / mcap_io.py
│   ├── single/{extract,convert,verify}.py
│   └── bimanual/{extract,convert,recovery,verify}.py
├── training/{train,config,adapters,bimanual,runtime}.py
├── configs/{single,bimanual}/
│   ├── conversion.json
│   └── pi0_lora.yaml / pi0_full.yaml / pi05_lora.yaml / pi05_full.yaml
├── scripts/                   # 変換、学習、圧縮、転送、コード共有、テスト
├── tests/
├── datasets/{mcap,lerobot}/
├── checkpoints/ / exports/ / logs/ / models/ / assets/
└── openpi/                    # 必要なソースのみをDockerと共有archiveに取り込む
```

初期コピーの `piper_pi0/` は参照資料として保持していますが、現在の変換・学習コードからは参照せず、Docker・共有archive・新規Git追加の対象から除外しています。データ・重み・ログもGit管理対象外です。

## 維持・移植した機能

- 双腕rawの監査、MCAP欠損復元、`recover`／`flag`／100ms、左7→右7の順序、front／left_wrist／right_wrist、無効画像の開始点除外と将来Actionの20Hzグリッド。
- 単腕raw→LeRobotの既存変換、front／wristの2カメラと7次元のstate／action。
- PiPER既存設定の検証・optimizer構築とOpenPIのtrainer、統計計算、Orbax保存・再開。
- 関節Actionの観測state基準delta化と、gripperの絶対値保持。

## 新しく実装・補完した機能

- 単腕・双腕を同じ `conversion/<robot>/` 配下に配置し、旧PiPER一式への実行時依存を除去。
- 両方のMCAP単体入力、複数MCAPの一括処理、共通ROS1 reader、実記録に基づくschema JSON。
- 単腕は同期許容値を厳密に検証。双腕は4つのrobot captureを30ms以内に合わせ、画像の有効性をquality列に保存。
- 共通学習CLI、8種類のYAML、モデル・LoRA／Full・GPU選択、構成確認、統計の一致検証、ログと有効YAMLの保存。
- チェックポイント選択・圧縮・manifest／SHA256検証・転送dry-run。コンテナで作ったメタデータをホストから使えるようパスを対応付け。
- 最小コード・設定・必要なOpenPI・ライセンスだけをまとめる共有archive。各ソースのSHA256も同梱。

## 確認済みの結果

| 対象 | 結果 |
|---|---|
| 既存環境 | `9219576bb4bd` のマウント元をinspectで確認。既存コンテナは停止・削除・設定変更せず、既存データ・重みに書き込まない |
| Docker | `sh build.sh` とCompose設定検証が成功。イメージは `tuning:local` |
| 依存関係 | Python 3.11.13、JAX 0.5.3、PyTorch 2.7.1、OpenPI、コミット固定LeRobot、MCAPをImport可能 |
| GPU | JAXとPyTorchでNVIDIA RTX A6000を2枚認識 |
| sim除外 | MuJoCo、gym-aloha、robosuite、dm-control、pymunk未導入。新イメージには `piper_pi0/` 一式、OpenPI examples／third_partyなし |
| 起動 | 初回起動、停止中起動、起動済み再利用、別ディレクトリからの起動、`exit` 後の稼働継続を確認 |
| Terminator | 起動引数と再接続をmockで確認。実ホストには未導入のためGUI起動は未検証。現在の端末での `sh` 接続は確認 |
| 単腕MCAP単体 | 実MCAP `20260825_224317` から503フレームを抽出し、LeRobotのParquet・2動画・fps・shapeを検証 |
| 双腕MCAP単体 | 実MCAP `episode_1792050571773182204` から1,416フレームを構成。3動画を含む全フレーム読込、画素比較、state／action、Action horizonを検証 |
| 既存rawの互換性 | 初回検証で単腕50エピソード／15,740フレームを入力検証。双腕rawの1,411フレームから欠損2を復元し1,413フレームを出力。移植後の同じ処理を回帰テスト48件で再確認 |
| 8種類の学習設定 | 単腕／双腕 × π0／π0.5 × LoRA／Fullの構築、モデルに対応したbase URI、実際のGemma variant・freeze filter、LoRA rank 16／32を確認 |
| 正規化 | 小規模の双腕LeRobotデータで実際に統計を計算。統計適用後のbatch、関節deltaの逆変換、padding、統計不一致の拒否、resume中の再計算拒否を確認 |
| 保存・再開 | CPU上の小さいNNXモデルとAdam optimizerを、OpenPIの実Orbax APIで保存・再開。step・重み・optimizer状態の一致を確認 |
| 圧縮・転送 | テストfixtureでarchive内部manifest、SHA256、上書き拒否、破損archiveの転送拒否、ホスト側パス変換を確認 |
| 独立した共有コード | 旧PiPER一式がない最小archiveの展開先で全62テスト成功。共有元の外部ソースを必要としないことを確認 |

MCAP単体の双腕1,416フレームと、同期済みraw由来の1,413フレームは、開始・終了範囲の決め方が異なります。MCAP単体ではrobot captureの共通時間範囲に20Hzグリッドを置きます。既存rawがある場合は元のグリッドを保持し、同じデータだと偽って一致扱いしません。

## 実行方法

```sh
sh build.sh
sh run.sh

sh scripts/convert.sh --robot single --dataset demo --input PATH --task 'Task text.'
sh scripts/convert.sh --robot bimanual --dataset demo --input PATH --task 'Task text.'
sh scripts/train.sh --robot bimanual --model pi05 --method lora --check-config
sh scripts/train.sh --config configs/bimanual/pi05_lora.yaml --norm-stats-only
sh scripts/train.sh --config configs/bimanual/pi05_lora.yaml --validate-only

# コンテナ内で、テスト用依存を/tmpにだけ配置して実行
sh scripts/test.sh -q

# ホストでも実行可能
sh scripts/package_project.sh
sh scripts/package_checkpoints.sh --experiment NAME --dry-run
sh scripts/transfer_checkpoints.sh --experiment NAME --destination user@host:/path --dry-run
```

実MCAP変換の検証出力は `/tmp/tuning-layout-validation`、今回のビルド・テストログは `logs/validation_20261025_layout/` に保存しています。以前の検証記録も同ディレクトリに保持しています。

## 未検証事項・制約

- 8種類のπ0／π0.5学習本番、収束、実モデルのVRAM使用量、実モデル重みの保存再開は未検証です。GPU上で継続中の既存作業には干渉せず、本番学習は開始していません。小規模fixtureの保存再開成功を実モデルの学習完了とは扱いません。
- 実データ全量での正規化統計の計算と、それを用いた学習は未実施です。小規模fixtureでは統計計算と正規化後の入力検証を実施しました。
- 別PCへの実SCP／rsync転送、転送先の実モデル復元、別GPU構成での再開は未検証です。転送先・認証情報はコードに固定していません。
- MCAPは確認済みのROS1レコーダー形式を対象にしています。別schema・ROS2・異なる単位や腕順序には追加調査とadapterが必要です。robot capture自体が欠落して同期グループを作れない場合、架空の値で埋めずに失敗します。
- Fullや双腕π0のYAMLは収束を保証しません。GPU数、batch、FSDPと必要なVRAMを調整してください。
- 固定したライブラリから動画API等の非推奨警告、型stub作成のImportWarning、Orbaxのsharding情報に関する警告が出ます。今回のテストは成功していますが、バージョン更新時は再検証が必要です。
- 別PCでの実ビルドは未実施です。公開base imageと固定依存、同梱の必要なソースから構築する定義で、旧 `piper-pi0-sim:111` をFROMやCOPY元に使用しません。

共有先ではDocker・Compose・NVIDIAドライバ・Container Toolkitを準備し、コードarchiveを展開して `sh build.sh` と `sh run.sh` を実行してください。教示データとbase model重みは別途配置またはダウンロードが必要です。
