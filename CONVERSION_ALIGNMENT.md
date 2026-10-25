# 単腕基準の変換仕様統一（2026-10-25）

単腕の数値変換・LeRobot writerを保持し、双腕を記録行保持のraw／独立最近傍のMCAPへ揃えました。単腕raw450・MCAP503は変更前と全Parquet列・標準metadata・全動画が一致しました。不規則な双腕rawは1,411行で変換できます。taskの簡素化と双腕画像の再選択廃止には、以下の意図的な差があります。

基準は作業前の安定状態を保存したGitコミット **`f8e8c1e273c57a28a05f3f84fa3cc9844ba7afd7`** です。`ec58a7a`へ単腕を再び書き戻す作業は行っていません。既存の比較出力は直前の単腕復元で基準コードとの完全一致を確認済みであり、今回もそれらと比較しました。旧ソースは `git show f8e8c1e:パス` で取得できます。

## 1. 変更前後の構造

変更前：

```text
conversion/
├── cli.py / pipeline.py
├── common/
│   ├── episode.py / mcap_io.py / raw_reader.py
│   └── synchronization.py / lerobot_writer.py / validation.py
├── single/
│   ├── adapter.py / convert.py / compatibility.py
│   └── extract.py / mcap_io.py / verify.py / diagnostics.py
└── bimanual/adapter.py
```

変更後（package入口は両腕とも維持）：

```text
conversion/
├── cli.py / pipeline.py
├── common/
│   ├── episode.py / mcap_io.py / raw_reader.py
│   ├── synchronization.py / lerobot_writer.py / validation.py
│   └── image_utils.py / fps.py / task.py / timing_diagnostics.py
├── single/
│   ├── adapter.py / convert.py / compatibility.py
│   └── extract.py / mcap_io.py / verify.py
└── bimanual/adapter.py
```

Adapterは名前・並び・カメラ定義、本体は入力解釈と出力処理です。単腕の保存API・EE処理を双腕writerへ統合せず、`convert.py` の名前も既存import保護のため維持しました。循環importは追加していません。従来の `grid_uses_cameras` はadapterの既存データ契約として保持しますが、新しい共通MCAP同期では常に全ストリームを使います。

## 2. 単腕の変更ファイル・関数

| 変更ファイル | 変更した関数 | 変更理由 | 出力への影響 |
|---|---|---|---|
| `conversion/single/convert.py` | `parse_args()`、`main()` | task簡素化、FPS helper、移動先import。置換用の未使用変数・tuple項目を整理 | taskは意図的変更。実raw/MCAP・EE比較の標準出力は一致 |
| `conversion/single/convert.py` → `common/image_utils.py` | `load_rgb()` | RGB/BILINEARを共有。関数のASTは基準と同一 | なし。全動画バイト一致で確認 |
| `conversion/single/extract.py` | `extract()` | metadata FPS/taskへ対応。既定20FPSの選択・JPEG quality95・SHA処理は保持 | 基準503行は全出力一致。別FPS指定時は明示仕様に従う |
| `conversion/single/diagnostics.py` → `common/timing_diagnostics.py` | `_numbers()`、`record_timing()` | 双腕でも同じ非破壊診断を使用。旧ファイルと不要ラッパーを削除 | 標準LeRobot出力への影響なし。警告ラベル・ログのFPS説明のみ更新 |
| `conversion/single/compatibility.py`、`verify.py`、`mcap_io.py`、`adapter.py`、直接CLI入口 | 既存関数すべて | 変更しない | ファイル内容一致 |

本体の関数位置：`conversion/single/convert.py:67` `parse_args()`、`conversion/single/convert.py:363` `main()`。MCAP抽出：`conversion/single/extract.py:20` `extract()`。

旧本体内でASTが完全一致する11関数は `load_json`、`load_yaml`、`resolve_path`、`correct_quaternion_continuity`、`quaternion_multiply_xyzw`、`quaternion_delta_rotvec`、`make_ee_delta_actions`、`episode_dirs`、`validate_episode`、`import_lerobot_dataset`、`infer_output_root` です。移動した `load_rgb` も基準関数とAST一致です。

削除したtask専用関数は `task_from_episode_meta`、`parse_task_replacements`、`apply_task_replacements`、`resolve_episode_task`。共通 `resolve_task` へ置換しました。State/Action演算、EE quaternion補正、N−1のEE差分、絶対Gripper、画像番号、LeRobot create/add_frame/save_episode/finalizeの呼び出し方は保持しています。

単腕ファイル群の差分は純追加13行、純削除164行、置換46旧行→20新行です。`git diff --numstat` では追加32行・削除209行（置換を含む）、差引177行減です。前述の純追加/削除/置換はdifflibによる区間分類で、Gitとは一致行の区切り方が異なります。物理行数と行置換の数え方を区別しています。詳細は `logs/spec_alignment/source-audit.json`。

## 3. 双腕の変更ファイル・関数

| ファイル | 関数・型 | 変更 |
|---|---|---|
| `common/raw_reader.py` | `read_episode` | rawの時計必須・20Hz間隔検査・capture再同期・カメラ再選択を廃止。CSV/NPZの行対応、画像実在、名前・単位・次元・有限値・実数の個数整合性は検査 |
| `common/raw_reader.py` | `read_episode` | complete/finalizedなどの追加ゲートを廃止。実際の配列/画像が揃う記録行を利用 |
| `common/synchronization.py` | `select_captures` | 全ストリーム共通範囲、採用FPS、独立最近傍、robot全体30ms。組合せ探索と未使用 `check_raw_grid` を削除 |
| `common/mcap_io.py` | `read_episode` | 共通の記録task/FPSを使用。ROS1型・名前・header・CRC検査は維持 |
| `common/episode.py` | `Episode` | 採用FPSを保持 |
| `common/lerobot_writer.py` | `write` | resize、採用FPS、episode間FPS一致、FPS付きsummary |
| `common/validation.py` | `check_episode`、`verify` | rawは1行以上。名目timestamp・動画・metadataを採用FPSで照合 |
| `pipeline.py` | `convert` | 共通引数、FPS整合、raw実時計ログ、検証後確定 |
| `cli.py` | `main` | 両腕共通 `--resize`・`--timing-log` |

根拠：`conversion/common/raw_reader.py:33` `read_episode()`、`conversion/common/synchronization.py:24` `select_captures()`、`conversion/common/lerobot_writer.py:18` `write()`、`conversion/common/validation.py:16` `verify()`。

`bimanual/adapter.py` 自体は未変更です。従来どおり左7→右7と3カメラの仕様を定義し、実処理は共通モジュールへ委譲します。

## 4. 共通化した処理と呼び出し元

| 共通処理 | 呼び出し元 | 保持した境界 |
|---|---|---|
| `image_utils.load_rgb/read_rgb` | 単腕converter、双腕writer | ファイルの旧RGB/BILINEAR処理を移動。MCAP bytesの入口も用意 |
| `fps.resolve_fps/consistent_fps/recorded_metadata` | 単腕converter/extract、共通MCAP/raw、pipeline | dataset優先 `int(fps)`、episode fallback、既定20と警告、採用FPS矛盾の拒否 |
| `task.resolve_task` | 単腕converter/extract、双腕raw/MCAP | 明示→episode→dataset→エラー。文字列はそのまま |
| `timing_diagnostics.record_timing` | 単腕converter、双腕pipeline | 時計なし/不規則を観察・警告するだけ。配列・画像・順序には介入しない |
| ROS1 decoder `mcap_io.read_streams` | 単腕の旧検査wrapper、双腕reader | 単腕の追加SHA/画像検査は残す |

単腕MCAPの短い既存最近傍ループ、単腕writer・verifyは無理に移動しません。双腕と同じ基本方式でも、JPEG中間生成・旧API互換・EE列と組み合わさる処理順の変更は出力を変える可能性があります。今回は補助機能だけを抽出しました。

## 5. 現行仕様の比較

| 項目 | 単腕 | 双腕 |
|---|---|---|
| raw | 記録行・連番JPEGを保持 | 記録行・CSV画像参照を保持 |
| 不規則時刻・時計なし | 許容、警告・実時計ログ | 同左。captureを使う再選択なし |
| 欠損/不正値 | 従来の配列・画像検査 | 配列・画像・CSV個数/ID整合性検査 |
| FPS | dataset `int(fps)` → episode → 20と警告 | 同左 |
| timestamp | 名目 frame_index/fps | 同左 |
| MCAP範囲 | 従来の全ストリーム共通範囲 | 今回同じ方式へ変更 |
| MCAP選択 | 独立最近傍、同距離は過去 | 同左、4系統の採用時刻最大差30ms |
| MCAP許容差 | robot各30ms、camera100ms、robot全体30ms | 同左 |
| resize | RGB後BILINEAR、未指定は元サイズ | 同左、3カメラ |
| task | 明示→episode task→dataset task→エラー | 同左 |
| State/Action | 7次元、従来順序・絶対関節/Gripper | 14次元、左7→右7・絶対関節/Gripper |
| カメラ | front/wrist | front/left_wrist/right_wrist |
| 固有機能 | V1/V5/V6、EE差分、include-failed、config/repo推論 | 学習互換 `quality.frame_valid=true` |

dataset FPSがある場合、episodeの実測FPSは矛盾判定や上書きに使いません。単腕の実rawはdataset20に対してepisode約17.85であり、20を保持します。dataset FPSがない場合に採用するepisode FPSが互いに違うとエラーです。

rawには30ms/100msの事後再同期保証を付けません。実時計をログへ残し、名目時刻と区別します。双腕valid列は学習互換のため受け入れた行にtrueを付けるもので、実時計が一定間隔だと宣言するものではありません。

## 6. コード行数

変換Pythonの物理行数（空行・コメント・package入口込み）。tests・検証scripts・設定・外部LeRobotは含めません。配置上の分類であり、双腕専用コードが10行だけという意味ではありません。双腕はcommonのreader/writerを利用します。

| 配置 | 基準 | 変更後 |
|---|---:|---:|
| common | 461 | 542 |
| single | 1,140 | 963 |
| bimanual | 10 | 10 |
| conversion直下 | 130 | 147 |
| 合計 | **1,741** | **1,662** |

79行減（約4.5%）。行数削減を優先した再実装は行っていません。

| ファイル | 基準行数 | 変更後行数 |
|---|---:|---:|
| `conversion/__init__.py` | 1 | 1 |
| `conversion/bimanual/__init__.py` | 1 | 1 |
| `conversion/bimanual/__main__.py` | 2 | 2 |
| `conversion/bimanual/adapter.py` | 7 | 7 |
| `conversion/cli.py` | 39 | 41 |
| `conversion/common/__init__.py` | 0 | 0 |
| `conversion/common/episode.py` | 45 | 46 |
| `conversion/common/fps.py` | 0 | 51 |
| `conversion/common/image_utils.py` | 0 | 41 |
| `conversion/common/lerobot_writer.py` | 64 | 65 |
| `conversion/common/mcap_io.py` | 86 | 87 |
| `conversion/common/raw_reader.py` | 134 | 87 |
| `conversion/common/synchronization.py` | 73 | 37 |
| `conversion/common/task.py` | 0 | 12 |
| `conversion/common/timing_diagnostics.py` | 0 | 57 |
| `conversion/common/validation.py` | 59 | 59 |
| `conversion/convert_mcap_to_lerobot.py` | 9 | 9 |
| `conversion/pipeline.py` | 81 | 96 |
| `conversion/single/__init__.py` | 1 | 1 |
| `conversion/single/__main__.py` | 3 | 3 |
| `conversion/single/adapter.py` | 7 | 7 |
| `conversion/single/compatibility.py` | 81 | 81 |
| `conversion/single/convert.py` | 814 | 691 |
| `conversion/single/diagnostics.py` | 57 | 0 |
| `conversion/single/extract.py` | 70 | 73 |
| `conversion/single/mcap_io.py` | 65 | 65 |
| `conversion/single/verify.py` | 42 | 42 |

## 7. 単腕の出力互換性

| 入力 | フレーム/episode | 全Parquet列・標準metadata | 全動画 |
|---|---|---|---|
| 実raw `20260825_224317` | 450/1 | 完全一致 | 2ファイルともバイト一致 |
| 同じMCAP、明示task | 503/1 | 完全一致 | 2ファイルともバイト一致 |
| EE pose＋resize＋失敗episode込みfixture | 8/2 | 完全一致 | 4ファイルともバイト一致 |
| EE差分＋resize＋失敗episode込みfixture | 6/2 | 完全一致 | 4ファイルともバイト一致 |

State・Action・timestamp・frame/episode/task index・task文字列・info・episode統計・画像対応を比較しています。追加provenanceログは標準LeRobotファイルと分けています。EE比較は旧置換結果の `Pick B`/`Pick D` を**新しい検証用fixtureの記録task**へ設定して比較しました。廃止した置換CLIが残っているとは主張しません。

実raw450の不規則間隔27か所、最大約150msは警告して記録します。実経過約25.15秒と出力名目終端22.45秒を区別し、行の生成・削除はありません。さらに元の50episode/15,740行を入力検証で受け入れました。50episode全動画の生成は行っていません。

証跡：`logs/spec_alignment/real/results.json`、`logs/spec_alignment/ee/ee-comparison.json`、`logs/spec_alignment/real.log`。

## 8. 双腕の実データ結果

| 入力 | 結果 | 確認 |
|---|---|---|
| raw `20261015_164839` | 1,096行、1episode | 14D左7→右7、3カメラ、20FPS、元サイズ |
| raw `20261015_165042` | **1,411行、1episode** | row837後の約150msを警告、2欠損行を追加せず成功 |
| 正常raw＋resize/明示task | 1,096行、1episode | 全3カメラ224×224、task一致 |
| 実MCAP10秒抜粋 | **198行、1episode** | 14D・3カメラ・最近傍同期・LeRobot reader成功 |
| 元MCAP全体 `164839` | 期待どおり拒否 | wrist1 row29、100.202ms >100ms。条件を緩めない |

正常rawの全Parquet列・info・episode情報は前後一致です。ただし旧実装はcapture時刻から画像を再選択していました。今回CSVに記録された画像へ戻したため、1,096行中front89行・left_wrist117行・right_wrist103行の参照が変わります。動画3本と画像統計が変わるのは意図した仕様変更です。例：front row6は旧 `front/000005.jpg` → 記録どおり `front/000006.jpg`。関節・task・timestamp・indexは変えません。

MCAP抜粋の旧範囲はrobot4系統で200行、新範囲は全カメラも含むため198行。始点が29.637813ms遅れ、50ms間隔の位相も変わります。旧新で完全一致するグリッド時刻は0件です。同じ新198ターゲットへ旧組合せsolverを適用すると、4robot・3cameraとも採用source index差は0件でした。これはこの実記録抜粋の結果であり、一般に同じ選択になる保証ではありません。代替の組なら成功しても独立最近傍のspanが30msを超えるケースはunit testで拒否を確認しています。

元MCAP全体は旧実装でもfront終端row1097の134.553msで拒否されていました。新仕様で別のcamera違反が先に検出されます。元データを書き換えたり、閾値を緩めたりして成功扱いにしていません。

証跡：`real/raw-camera-selection.json`、`real/dual-baseline-comparison.json`、`real/mcap-same-targets.json`、`mcap-precheck.json`（すべて `logs/spec_alignment/` 配下）。抜粋ファイルは既存の実MCAP抜粋を利用し、payloadを生成・修正していません。

## 9. π0／π0.5学習互換性

実出力6経路×2モデルの**12組**で、変更していない既存学習コードのdataset検査と実batch読み込みが成功しました。単腕state `(1,32)`/action `(1,20,32)`、双腕state `(1,32)`/action `(1,50,32)`、画像224×224。7D/14Dの関節Action delta化、Gripper絶対値、逆変換、padding、カメラkey、双腕valid列を検証しています。

実batchは正規化前の検証です。正規化と往復処理は小さなfixtureの回帰テストで確認しています。モデルの重み読み込み・学習・収束試験は実行していません。EE差分は従来どおり関節学習用Actionとは別表現です。

10FPSのfixtureではmetadata・timestamp・動画FPSの一致とLeRobot reader読み込みを両腕で確認しました。既存双腕学習の `training/bimanual.py:34` は20FPSを要求します。非20FPSの学習対応を今回追加したとは主張しません。

## 10. 利用できなくなった機能・意図的変更

- 単腕 `--input-task`、`--task-replace` とtask多段置換を削除。未対応引数として明確に拒否します。
- `tsku` alias、task/tsku矛盾検査、YAML task fallback、暗黙の汎用文言を廃止。必要なら `--task` または記録taskを用意します。
- task文字列はstripせず保持します。従来の前後空白除去と異なり、空白がある文字列は意図的に変わります。空白だけのtaskは受け入れません。
- 双腕MCAPの4系統組合せ探索・robotだけのグリッドを廃止。変換長・位相・成功率が変わり得ます。
- 双腕rawのcapture再同期・画像再選択・50ms強制を廃止。記録画像へ戻るため旧動画と差が出る場合があります。

EE差分、raw形式、入力統合、失敗episode選択、resize、configの出力/repo推論、直接CLIの明示overwriteは維持します。高度な復元・quality flag・全画像SHA照合・全フレームreader比較を標準変換に戻していません。

## 11. 残る制限と保護対象

元のMCAP/rawはread-only bind mountで検証し、既存datasetを上書きしていません。出力は `logs/spec_alignment/` の新規領域です。学習コード、学習済みチェックポイント、`piper-pi0-sim:111`、既存Docker環境は変更していません。既存コンテナを停止・再作成せず、既存イメージを使う一時検証コンテナだけを作成・自動削除しました。Docker buildは実行していません。

`training/` 全ファイル、`build.sh`、`run.sh`、Dockerfile、compose.yaml、`scripts/convert.sh` は基準との内容一致をSHA256付きで記録しました。イメージ内 `/opt/project_minimal` は既存依存ソース、`/workspace/project_minimal` はbind mountの今回のコードです。公開入口は後者を使うため今回の利用に再ビルドは不要です。

rawの画像や配列自体が欠損している場合は依然エラーです。時計だけの欠損・不規則とは区別します。単腕rawの形式情報を持つdataset metadataの要件は保持しています。FPS値だけがない場合のepisode fallbackと、形式情報そのものがない入力は別です。

`--validate-only` はdatasetを生成しませんがraw時刻ログを新規作成します。単腕MCAPでは従来どおり `<output>.raw/` の抽出も行います。共有入口は上書き禁止・partial検証後rename、直接単腕CLIの明示overwriteは従来仕様です。

## 12. テスト・ドキュメント

- 回帰テスト52件成功：同期境界・同距離過去側・独立選択span超過・不規則raw/時計なし・finite/名前/型/画像/shape不正・task/旧引数拒否・FPS優先/矛盾/既定・10FPS/resize・原子的な確定・8学習設定・正規化・小規模Orbax fixture。
- 実変換6経路とLeRobot reader、12学習batch、単腕50episode入力検証、EE比較2経路が成功。
- 公開shellは単腕raw/MCAP、双腕正常/不規則raw、双腕MCAP抜粋の入力検証が成功。
- 実MCAP全体の100ms超過、unit testの独立最近傍span超過・欠損・不正値・破損画像は期待する拒否です。
- 最初の検証スクリプト起動ではOpenPI側のscripts packageとの名前衝突を修正しました。双腕への「全metadata一致」検査は画像統計差を検出し、調査で記録画像保持による意図した差と特定しました。隠して一致扱いにはしていません。

検証コード：`scripts/verify_conversion_alignment.py`、`tests/test_conversion_alignment.py`。実データ比較scriptは新規 `--output` と既存の `--baseline` を要求し、Git基準solverも参照します。今回の実行は実データ本体・EE比較・同一グリッド比較を段階的に実施し、各JSONへ保存しました。回帰ログは `logs/spec_alignment/tests-clean.log`、公開CLIは `real/cli.json` と `cli-dual-mcap.log`。

READMEを現行の両腕共通resize/task/FPS、rawとMCAPの違い、時刻ログ、制限へ更新しました。`conversion/README.md` に責務と呼び出し元を記載し、旧復元報告は履歴と明記しました。共有packagerへ本報告を追加します。既存共有archiveは上書きしません。
