> この文書は旧仕様復元時点の記録です。その後の明示的なtask簡素化・双腕仕様統一は [CONVERSION_ALIGNMENT.md](CONVERSION_ALIGNMENT.md) を参照してください。

# 単腕変換の旧仕様復元（2026-10-25）

単腕を双腕用の厳密なraw受け入れ条件へ合わせた変更を取り消し、`ec58a7a` の単腕converterへ戻しました。指定の `20260825_224317` rawは450フレームで変換でき、基準コミットとState・Action・timestamp・task・全メタデータ・2動画が完全一致しました。双腕の簡素化経路、学習コード、元データ、チェックポイント、既存Dockerイメージ・コンテナは変更していません。

## 1. 比較したコードと復元方法

| 対象 | ec58a7a | 604行版での変更 | 今回の対応 |
|---|---|---|---|
| `single/convert.py` | 804行、18関数 | 削除、共通reader/writerで再実装 | 復元。814行。16関数はAST一致、2関数はargv・時刻ログ追加のみ |
| `single/extract.py` | 70行 | 中間rawを使わないMCAP経路へ変更 | 復元。関数本体はAST一致。readerのimport先だけ変更 |
| `single/verify.py` | 42行 | 共通validationへ変更 | 基準ファイルとバイト単位で一致 |
| 旧共通MCAP reader | 100行 | 画像SHA・画像verify等を削減 | ROS1 decoderは共用し、単腕に旧schema・ファイル選択・SHA・画像verifyを復元 |
| 共有CLI | 単腕追加引数を旧parserへ転送 | 追加引数を拒否 | 単腕だけ旧parserへの転送を復元 |
| `python -m conversion.single` | 旧converterの直接入口 | dataset必須の共通CLIへ変更 | 旧直接入口へ復元 |

`convert.py` の変更関数は `parse_args()` と `main()`。変換演算・選択・task・設定推論の16関数は基準と同一です。証跡は `logs/single_compatibility/function-diff.json`。

共通入口は `conversion/cli.py` → `pipeline.py` を維持します。単腕は `single/compatibility.py` → 旧converter、双腕は従来の `common/` → 共通writerを呼びます。単腕の出力互換性を優先し、writerまで無理に統一していません。

## 2. 失われていた機能・受け入れ条件

| 項目 | 旧単腕仕様 | 604行版 | 復元後 |
|---|---|---|---|
| 不規則なraw時刻・欠損間隔 | 記録行を保持して名目timestampを付ける | 拒否 | 受け入れ、実時計と警告を別ログへ記録 |
| rawの時計なし | State/Action・画像があれば可 | timestamp必須 | 受け入れ、実時計がないことを警告 |
| rawのCSV | 変換の入力契約に含まない | frame ID・時刻・画像参照へ使用 | 旧どおり参照せず連番JPEGを使用 |
| rawのcapture時刻 | 並べ替え・再同期・許容差検査には使わない | 許容差で拒否し、画像を再選択 | 元の行と画像対応を保持。時計はログだけに記録 |
| rawのFPS | dataset metadataの `int(fps)`、既定20 | 20固定、episode configured FPSも検査 | dataset metadataのFPSへ復元 |
| episode実測FPS | 出力FPSの決定には使わない | 条件により拒否 | 旧どおり無視。実時刻差は別記録 |
| raw形式 | V1 / V5 / V6 EE poseを明示対応 | 元のformat識別なし | 3形式の対応と未知format拒否を復元 |
| State/Action・名前 | metadataのdim/namesを検査し複数入力間でも照合 | canonical値を強制 | 元のmetadata解釈と入力間整合性検査を復元 |
| episodeの選択 | `successful=false` を既定除外、指定時に含める | 追加の完了状態・個数検査 | 旧選択に復元。complete/finalized等を理由に追加拒否しない |
| `num_frames` | 実際のNPZ配列・画像数を使う | 宣言値との一致を必須化 | 元のNPZ/画像基準へ復元 |
| 1フレームraw | 関節形式で受け入れ可能 | N>=2を必須化 | 旧条件へ復元。EE差分は従来どおり最低2行 |
| task未指定 | 優先順位と既定文言で解決 | 欠落時に拒否 | 旧優先順位・既定文言を復元 |
| task文字列 | strip、tsku互換、矛盾検出、置換 | 一部を削除・文字列保持へ変更 | 全て復元 |
| EE pose | optional 8D feature、quaternion連続性補正 | 削除 | 復元 |
| EE差分Action | 位置差＋回転ベクトル差＋次時刻の絶対Gripper、N−1行 | 削除 | 復元 |
| resize | RGB読み込み後のBILINEAR resize | 削除 | 復元 |
| MCAP中間JPEG | quality95のrawを経由 | 元圧縮画像から直接writerへ | quality95の旧処理へ復元 |
| 一時PNG・LeRobot API | 旧create/add_frame/save_episode、既定の画像保存 | 無圧縮PNG writerへ変更 | 単腕は旧API・既定の画像保存へ復元 |
| LeRobot import/finalize互換 | 3つのimport候補、finalizeがあれば呼ぶ | 固定APIのみ | 旧本体の互換処理を維持 |

State/Actionの形状・有限値、画像枚数・連番ファイル等の旧検査は残します。「不規則時刻を受け入れる」は、画像欠損や不正な関節値まで受け入れるという意味ではありません。共有入口の基本出力検証も旧 `verify.py` で実行します。

### 旧オプションの対応

| オプション | 復元内容 |
|---|---|
| `--input` | 繰り返し指定してrawを統合、互換schemaを照合 |
| `--input-task` | 各inputと同じ順・個数でtaskを指定 |
| `--output` | 明示出力先。共有入口はpartial後に確定 |
| `--config` | YAMLのtask fallback、直接CLIでは出力先・repo推論 |
| `--repo-id` | 明示指定→設定→dataset metadata→既定値の優先順位 |
| `--task` | 全体のtask指定 |
| `--task-replace` | OLD=NEWを順に適用、複数指定可 |
| `--action-mode` | joint_absolute / ee_delta_axis_angle |
| `--resize WIDTH HEIGHT` | 正のサイズへ旧補間方式でresize |
| `--include-failed` | successful=falseも選択 |
| `--overwrite` | 旧直接CLIで維持。共有shellは基準と同じく禁止 |
| `--validate-only` | rawは旧入力検証のみ。LeRobotをimport・生成しない。時刻ログは新規追加 |

追加したオプションは `--timing-log PATH` だけです。既存のログは上書きせず、新しいパスを要求します。

## 3. 実測時刻と出力timestampの区別

`single/diagnostics.py` は変換に使う配列・画像・順序を変更しません。

- 既定出力は従来どおり `frame_index / dataset_metadata_fps`。
- 警告は、不規則区間数・先頭の実時刻差・名目周期・完全ログの場所を表示。
- `<output>.raw_timing-UUID.json` に元timestamp、全隣接時刻差、各不規則区間の行番号・実測delta・名目deltaを保存。
- 完成データにも `provenance/raw_timing.json` を保存。
- 時計が欠落・不正な形式・非有限値でも、旧converterが受け入れるデータを時計だけで追加拒否しない。監査不能部分は警告し、非有限値はJSONのnullで区別。
- ログの出力時計には `nominal ... NOT measured capture time` を明記。

指定raw450行では、`timestamp` に27か所の不規則間隔があり、最大約150msでした。実時計の先頭から末尾は約25.15秒、出力名目timestampの末尾は449/20=22.45秒です。この差を隠さず記録し、出力自体は旧版と同じにしています。欠損行の生成・ゼロ埋め・補間・分割は行いません。

実ログ：`logs/single_compatibility/real/restored_raw_450/provenance/raw_timing.json`。

## 4. 変更前後の出力比較

同じ既存イメージと固定LeRobotを使い、元データをread-onlyマウントした一時検証コンテナで比較しました。基準はGitから取得した旧コードです。MCAPの基準抽出も旧reader・旧extract関数で実施し、復元後とは独立して出力しています。

| 入力 | フレーム / episode | State・Action・timestamp・task等の全Parquet列 | info / episodes / tasks / episode統計 | 全動画 |
|---|---|---|---|---|
| 実raw `20260825_224317` | 450 / 1 | 完全一致 | 完全一致 | 2ファイルともバイト一致 |
| 同じ実MCAP | 503 / 1 | 完全一致 | 完全一致 | 2ファイルともバイト一致 |
| EE絶対Action＋resize＋入力別task＋task置換＋失敗episodeを含むfixture | 8 / 2 | 完全一致 | 完全一致 | 4ファイルともバイト一致 |
| EE差分Action＋同オプションfixture | 6 / 2 | 完全一致 | 完全一致 | 4ファイルともバイト一致 |

動画の標本比較だけでなく、全動画ファイルの一致を確認しています。追加した時刻ログ・provenanceは比較対象の標準LeRobotファイルとは別です。

さらに元の単腕raw 50エピソードを旧版・復元後の両方で `--validate-only` に通し、両方で受け入れを確認しました。50件全ての動画生成は実施していません。

証跡：`logs/single_compatibility/real/comparison.json`、`acceptance.json`、`real-comparison.log`。再現スクリプトは `scripts/verify_single_compatibility.py`。参照ソースは `git archive ec58a7a conversion configs` から一時ディレクトリへ展開して指定します。計測用の旧ソース複製は検証後に削除し、共有パッケージには含めません。

## 5. 回帰テストと双腕の維持

- 全回帰テスト44件成功。
- 不規則rawの保持と実時計ログ、時計なし、V1/V5/V6、1行raw、EE差分、quaternion、resize、task/tsku、入力別task・置換、失敗episode選択、設定推論、共有CLI転送、指定ログの非上書きを確認。
- 双腕の20Hz/30ms/100ms・4系統整合性・欠損拒否・画像欠損/shape/不正値・非公開partialの既存テストは継続成功。
- 公開CLIは単腕の通常raw・MCAP・不規則raw、双腕の通常raw・MCAPで成功。双腕の欠損rawと許容差超過MCAPは従来どおり非ゼロ終了。
- `training/` と `conversion/common/` 全Python・双腕adapterは、今回の作業前後のSHA256が一致。双腕の変換本体と学習コードは変更していません。

共通pipelineの変更は単腕分岐と単腕オプション転送だけです。双腕へ復元・分割・flag・全画像比較を追加していません。既存の学習済みチェックポイントを使う学習や再学習は実行していません。

## 6. 残る差異・注意点

1. **時刻ログが新しく追加されます。** raw `--validate-only` でもログを新規作成します。LeRobotの標準列・画像・metadataは旧出力のままです。
2. **単腕MCAPの中間raw保存先は `<output>.raw/`。** 旧共有CLIの `datasets/extracted/<dataset>` とは異なります。抽出内容は同じです。既存中間rawは上書きしません。
3. **episode単体の便利な指定を残しています。** `data.npz` があるディレクトリはrawを優先し、親のdataset_metaを一時ラッパーへ読み込みます。旧共有CLIはdataset_metaが直下にないepisodeをMCAP入力と判定していました。MCAPとrawが同居するepisodeでMCAPを変換する場合は、`.mcap` ファイル自体を指定してください。旧raw本体と同じ入力判定にする場合はdataset_metaがあるrawルートを指定します。
4. **共有入口と直接入口の違いは基準どおり。** 共有CLIはdataset名を基に出力・repoを渡し、上書きを禁止し、検証後に確定します。YAMLから出力先を自動推論する場合は `python -m conversion.single --input ... --config ...`。直接入口の明示overwriteや書き込み方法は旧仕様です。
5. **EE差分Actionは関節Actionと別表現です。** 既定の関節学習用にはjoint_absoluteを使います。学習側の関節delta化・絶対Gripperは変更していません。
6. **単腕の以前の速度改善値は撤回します。** JPEG中間変換・旧writerを復元しているため、604行版の単腕速度は現行には適用できません。今回のraw450では旧74.71秒、復元後73.45秒でしたが、単回・検証範囲も異なるため速度向上は主張しません。
7. **旧piper_pi0側の重複ソースは再導入しません。** 単腕の正本は `conversion/single/`。共有shell入口と `python -m conversion.single` を維持します。

READMEとconversion/READMEを現行仕様へ更新し、`CONVERSION_SIMPLIFICATION.md` は単腕復元前の履歴と明示しました。Dockerビルド・イメージ更新・既存コンテナ再作成・依存ライブラリ更新は行っていません。
