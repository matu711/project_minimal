# ソースの由来とライセンス

本リポジトリは、PiPER実機用のデータ変換・Fine-tuningと、同梱OpenPIの必要部分で構成します。元ソースの著作権・ライセンスヘッダーを保持してください。操作手順はルートのREADME.mdを参照してください。

## PiPER由来の処理

単腕変換は、旧PiPER実装との出力互換性を維持するため、復元した処理を使用しています。双腕変換は共有pipelineとrobot別adapterを使用します。以下は初期移植元の記録で、SHA256は現在の編集済みファイルのhashではありません。

| 現在の配置 | 初期移植元 | 初期移植元SHA256 |
|---|---|---|
| `conversion/single/convert.py` | `piper_pi0/real/collect/convert_mcap_to_lerobot.py` | `651272f9df5a748d0db91d4f6a3c1ee989ebac850846335c432e746f02fdedec` |
| `training/config.py` | `piper_pi0/training/scripts/train_bimanual_pi05.py` | `bcbe00d37710168d6ebcb8700dc9c45216d31223964ae51db222133071f1adf2` |
| `training/bimanual.py` | `piper_pi0/training/scripts/train_bimanual_pi05.py` | `bcbe00d37710168d6ebcb8700dc9c45216d31223964ae51db222133071f1adf2` |
| `training/adapters.py` | `piper_pi0/training/configs/bimanual.py` | `29230b1b8ab43d5cdf818749314add5dfa0d9b87fff513bbdb3149c67cd44bd2` |

移植元のPiPERツリーには独立したLICENSEファイルがありませんでした。本リポジトリから新しい許諾条件は付与せず、既存の出典・docstringを保持しています。移植元パスは由来の説明であり、実行時の依存先ではありません。

## OpenPIとモデル

`openpi/` は同梱snapshotです。上流のApache-2.0ライセンスは `openpi/LICENSE`、Gemmaの利用条件は `openpi/LICENSE_GEMMA.txt` に保持しています。モデル内のBig Vision／Google等の著作権表記も保持してください。第三者表記の一覧は `THIRD_PARTY_NOTICES.md` を参照してください。

モデル重み・tokenizer・データセットはソース配布に含めません。取得・利用・再配布には各資産の適用条件が別途あります。

## 依存と整合性

LeRobotの依存commitとPythonライブラリは `requirements.lock` で固定します。Checkpointパッケージには `MANIFEST.sha256.json` とarchive外部の `.sha256` を保存します。ソース履歴はGitで管理し、別のソース包装スクリプトは必要としません。
