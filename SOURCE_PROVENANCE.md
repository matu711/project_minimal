# ソースの由来とライセンス

初期コピーのPiPER・OpenPIから、変換と学習に必要な部分をコピーしました。下表は初期移植時の履歴です。変換コードはその後共通pipelineへ再実装しましたが、単腕は互換性のため `ec58a7a` のconvert/extract/verifyを復元しました。双腕の旧converter・復元・詳細検証と旧テストは同コミットに保存して削除しています。
共有用パッケージには実行に必要なソースだけを含め、元のPiPER一式・sim・archiveは含めません。

| 新しい配置 | コピー元 | コピー元SHA256 |
|---|---|---|
| `conversion/single/convert.py` | `piper_pi0/real/collect/convert_mcap_to_lerobot.py` | `651272f9df5a748d0db91d4f6a3c1ee989ebac850846335c432e746f02fdedec` |
| `conversion/bimanual/convert.py` | `piper_pi0/real/collect/convert_bimanual_to_lerobot.py` | `73890e3e3daa384e5072fbd0affdff502d9fb6cdc7404ef3b5178a98e52d0fe1` |
| `conversion/bimanual/verify.py` | `piper_pi0/real/collect/convert_bimanual_to_lerobot.py` | `73890e3e3daa384e5072fbd0affdff502d9fb6cdc7404ef3b5178a98e52d0fe1` |
| `conversion/bimanual/recovery.py` | `piper_pi0/real/collect/bimanual_mcap_recovery.py` | `85f5d37e2e812a7c71227caa3622fa82d46ed17a3a1b10e2b9e4ecc482d62840` |
| `training/config.py` | `piper_pi0/training/scripts/train_bimanual_pi05.py` | `bcbe00d37710168d6ebcb8700dc9c45216d31223964ae51db222133071f1adf2` |
| `training/bimanual.py` | `piper_pi0/training/scripts/train_bimanual_pi05.py` | `bcbe00d37710168d6ebcb8700dc9c45216d31223964ae51db222133071f1adf2` |
| `training/adapters.py` | `piper_pi0/training/configs/bimanual.py` | `29230b1b8ab43d5cdf818749314add5dfa0d9b87fff513bbdb3149c67cd44bd2` |
| `tests/test_bimanual_conversion.py` | `piper_pi0/real/collect/test_convert_bimanual_to_lerobot.py` | `233b4ee97cc07f3efae1e26e9b391cf6ca638e2a8440fdfa0ffd6e96834dccbf` |
| `tests/test_bimanual_recovery.py` | `piper_pi0/real/collect/test_bimanual_mcap_recovery.py` | `9409d1169c8c0a1c15caa4c2310ccbfca555c38ab1bc7839455787fcf8a685f0` |

現在の双腕変換は `conversion/common/` とrobot別adapterを使います。単腕は復元した旧converterを使用し、argv受け取りと実時刻の警告・ログだけを追加しています。双腕の4系統選択規則は保持し、raw復元・分割・詳細quality・全画像照合は提供しません。学習コードは変更していません。
学習設定の構築はモデル・robot・学習方法の選択と共有用パスを追加し、不要な旧pipelineへの依存を除きました。
単腕adapterは既存 `train_real_pi0.py`／`train_real_pi05_pbl1_joint.py` と同じtransform構成です。
MCAP単体の抽出・共通reader・各ロボットのschema JSON・共有CLI・パッケージ処理は新規実装です。

OpenPIは同梱snapshotを使用します。元のライセンス `openpi/LICENSE` と `openpi/LICENSE_GEMMA.txt` を保持しています。
PiPERコピー元には独立したLICENSEファイルがないため、新しい許諾条件は付けていません。元の表記・docstringを保持しています。
LeRobotの依存コミットは requirements.lock に固定しています。
共有archive内の `SHARED_SOURCE_SHA256.json` は実際に含めた各ソースのhashです。
