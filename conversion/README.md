# MCAP / raw → LeRobot

公開入口は `cli.py` → `pipeline.py`。単腕の安定した数値変換・writerを保持し、双腕は簡素なreader/writerを使用します。ファイル数を揃えるための単腕再実装は行いません。

```sh
sh scripts/convert.sh --robot single --dataset example --input /path/to/raw --resize 224 224
sh scripts/convert.sh --robot bimanual --dataset dual --input /path/to/raw --resize 224 224
sh scripts/convert.sh --robot bimanual --dataset dual_mcap --input /path/to/record.mcap --task 'Assembling a box.'
python3 -m conversion.single --help
```

## 責務と呼び出し関係

- `single/adapter.py`：単腕の名前・順序・カメラ定義。
- `single/compatibility.py`：共有CLIの出力保護、episode単体指定、MCAP抽出、本体への接続、検証後rename。
- `single/convert.py`：V1/V5/V6検査、episode選択、EE姿勢・EE差分、従来のLeRobot create/add_frame/save_episode/finalize。関節値・画像選択・名目timestamp生成を保持。
- `single/extract.py`：MCAPの独立最近傍同期と従来のquality95 JPEG中間raw。metadata FPSとtask解決を共通helperに委譲。
- `single/mcap_io.py`：共通ROS1 decoderの上で既存の画像verify・SHA256検査を維持。
- `single/verify.py`：従来の基本Parquet・動画検証を変更せず使用。
- `bimanual/adapter.py`：左7→右7、front/left_wrist/right_wristなど双腕固有定義。
- `common/raw_reader.py`：双腕NPZ・CSV画像参照の行対応を保持。記録右7→左7を出力順へ変更。実時計による拒否・画像再選択は行わない。
- `common/mcap_io.py`：両腕で共有する型・名前・単位・header検査。双腕のMCAP Episode構築も担当。
- `common/synchronization.py`：全ストリーム共通範囲のFPSグリッド、過去側優先の独立最近傍。robot各30ms、camera100ms、選択robot全体30ms。単腕extractの既存選択ループは出力保護のため残す。
- `common/lerobot_writer.py`：双腕の絶対Action、動画、互換列 `quality.frame_valid=true`。
- `common/validation.py`：双腕の有限値・次元・フレーム数・採用FPS・動画shapeの検証。
- `common/image_utils.py`：旧単腕 `load_rgb` を演算順・例外・戻り値を変えず移動。双腕のbytes画像も同じRGB→BILINEAR処理。
- `common/fps.py`：dataset `int(fps)` → episode → 20（警告）。採用FPS矛盾はエラー。dataset FPSがある場合はepisode実測FPSで上書きしない。
- `common/task.py`：`--task` → episode task → dataset task → エラー。文字列はそのまま。置換・入力別task・tsku・YAML task fallback・既定文言なし。
- `common/timing_diagnostics.py`：実測時計と全隣接時刻差のログ。配列・行順・画像対応を変更せず不規則時刻を警告。

rawは両腕とも記録行を保持します。欠損行の追加・復元・分割・再同期はありません。単腕は従来どおり連番JPEG、双腕は記録CSVの画像参照を使用します。名目 `frame_index / fps` と実測時計を区別し、時計のないrawでも警告して既存行を利用します。画像欠損、壊れた画像、State/Action欠損・非有限値・次元不正は拒否します。

MCAPは厳密同期を要求します。schemaのfps=20は既存recorderの識別条件であり、出力グリッドFPSは記録metadataから解決します。単腕20FPSの既存選択・JPEG生成・writer出力を維持します。双腕の以前の組合せ探索やロボットだけのグリッド範囲は廃止しました。

`--resize WIDTH HEIGHT` は全カメラへ適用します。未指定なら元サイズ。`--task`、`--timing-log PATH`、繰り返し `--input`、`--output`、`--mcap-schema`、`--validate-only` は共有入口で利用できます。単腕ではさらに `--config`、`--repo-id`、`--action-mode`、`--include-failed` を旧parserへ転送します。`--overwrite` は単腕直接CLIだけの明示機能です。

共有入口は上書きを禁止し、新規partial→基本検証→renameで確定します。raw `--validate-only` も時刻ログを新規作成します。単腕MCAPは `<output>.raw/` の抽出も行います。

学習処理は変更していません。EE差分Actionは関節学習と別表現です。双腕の既存学習入力検査は20FPSを要求します。別FPSの変換出力の読み込み確認はLeRobotまでです。詳細は [CONVERSION_ALIGNMENT.md](../CONVERSION_ALIGNMENT.md) を参照してください。
