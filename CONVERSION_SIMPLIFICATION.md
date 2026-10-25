> この文書は単腕互換性の復元前の履歴です。現在の単腕は旧converterへ戻しており、以下の「欠損raw拒否」「604行」「単腕の速度改善」は現行単腕には適用されません。現行の仕様・比較は [SINGLE_COMPATIBILITY_RESTORE.md](SINGLE_COMPATIBILITY_RESTORE.md) を参照してください。双腕の簡素化経路は維持しています。

# 単腕・双腕LeRobot変換の簡素化結果（2026-10-25）

単腕・双腕の標準変換を同じpipelineへ統一しました。変換用Pythonコードは2,356行から604行へ74.4%削減し、実データの4経路は単回実測で61〜77%短縮しました。学習コード、元データ、学習済みチェックポイント、既存イメージは変更していません。

## 1. 削除・簡素化した機能

- raw欠損のMCAP復元、recover、欠損でのepisode分割、image-policy flag。
- 詳細quality/source列、画像SHA256全件照合、各段階の入力全件ハッシュ。
- 全フレームのLeRobot reader読み直し、画素比較、詳細な複数時計のdrift監査。
- 復元専用Episode情報、segment管理、旧単腕EE差分Action・resize等の個別CLI機能。
- `conversion/` の旧extract/convert/recovery/verifyと対応する旧テスト、`piper_pi0/real/collect/` の旧converter・復元・変換専用検証・テスト。

元はGit管理されていなかったため、削除前にソースをGitコミット `ec58a7a`（`Record source baseline before conversion simplification`）へ保存しました。削除した各ファイルは `git show ec58a7a:パス` で参照できます。計測用の旧ソース複製は測定後に削除し、共有物にも含めません。旧学習・実機制御の参照資料全体を削除したわけではありませんが、それらは従来どおりDocker・共有パッケージの対象外です。

## 2. 共通化した機能

```text
conversion/
├── cli.py
├── pipeline.py
├── convert_mcap_to_lerobot.py    # 既存shell用の互換入口
├── common/
│   ├── episode.py               # 小さなEpisode・RobotSpec・工程時間
│   ├── mcap_io.py               # ROS1型・名前・header検証、MCAP読み込み
│   ├── raw_reader.py            # NPZ・JSON・CSV・画像の読み込み
│   ├── synchronization.py       # 20Hz、最近傍、許容誤差、raw欠損拒否
│   ├── lerobot_writer.py        # 絶対Action、共通features・動画生成
│   └── validation.py            # Parquet・動画・基本metadata検証
├── single/adapter.py
└── bimanual/adapter.py
```

共通条件は20Hz、robot 30ms、camera 100ms、線形補間なし、同期不能時はエラーです。双腕の4ロボットストリームは互いに30ms以内の組を選び、従来の選択スコアと同率時の探索順も維持しました。

rawは既に同期済みの行を利用し、20Hz間隔・連続frame ID・配列・画像対応を検査します。capture時刻があれば実時刻の許容差も確認します。rawにMCAPが同居していても開かず、自動復元しません。

出力は新規 `.名前.partial-UUID` に作成し、基本検証が成功してから正式名へrenameします。既存出力の上書きを禁止し、失敗したpartialを完成データとして公開しません。

中間PNGは圧縮レベル0、動画は同じ固定LeRobotの既定AV1です。MCAPから中間raw/JPEGを書き出す工程をなくし、選択画像を直接writerでデコードします。

## 3. 単腕・双腕で残した違い

| 項目 | 単腕 | 双腕 |
|---|---|---|
| State / Action | 7次元、現行順 | 14次元、左7→右7 |
| カメラ | front / wrist | front / left_wrist / right_wrist |
| 入力topic | 現行単腕topic | 現行左右4topic |
| MCAPグリッド範囲 | ロボットとカメラ全体の共通範囲 | ロボット4系統の共通範囲 |
| ロボット同期 | 独立最近傍＋State/Action間30ms | 4系統の組を選択し全体30ms |
| raw | dataset_meta・NPZ・meta・画像、CSVがあれば参照 | NPZ・meta・CSV・画像、完了情報 |
| 学習互換列 | なし | `quality.frame_valid=true` 1列のみ |

グリッド範囲の違いはAdapter設定として残しました。正常な比較データでは変更前後のフレーム数・episode数は同じです。単体テストでも、カメラ開始が遅いケースの単腕3行／双腕4行を維持することを確認しています。

絶対関節Action・絶対Gripperを保存します。学習時の関節delta化とGripper絶対値、逆変換は変更していません。双腕の既存学習がvalid列を必須とするため、詳細qualityをなくしつつ互換用の1列を残しています。入力が不正なら、この列をtrueにして出力する前に拒否します。

## 4. コード行数

変換用Python全ファイルの物理行数です。空行・コメント・`__init__`・`__main__`・互換入口を含み、テスト・計測スクリプト・設定JSON・外部LeRobotは含みません。旧コピーを二重計上していません。

| 区分 | 変更前 | 変更後 |
|---|---:|---:|
| 共通 | 198 | 584 |
| 単腕専用 | 920 | 10 |
| 双腕専用 | 1,238 | 10 |
| **合計** | **2,356** | **604** |

合計1,752行、74.4%削減しました。行数を減らすために同期・型・画像欠損・有限値・基本出力検証を削ってはいません。

## 5. 変更前後の変換時間

同じ既存イメージ `b7567ab8…`、Python 3.11、同じ固定LeRobot、同じAV1 codec、CPU上限6・メモリ上限16GBで順次実行しました。元データはread-only bind mount、結果は `logs/conversion_simplification/` の新規ディレクトリです。既存サービスは停止・再作成していません。

共通ライブラリの事前import後から計測し、読み込み・同期・画像処理・動画生成・基本/旧詳細検証までを含みます。各条件1回で、OSキャッシュを強制消去していません。改善率はこの測定条件の結果で、一般的な速度保証ではありません。

| 経路 | フレーム | 前：秒 | 後：秒 | ms/frame：前→後 | 時間短縮 |
|---|---:|---:|---:|---:|---:|
| 単腕raw（既存MCAP抽出済み） | 503 | 82.358 | 19.249 | 163.73 → 38.27 | 76.6% |
| 単腕MCAP | 503 | 84.530 | 19.864 | 168.05 → 39.49 | 76.5% |
| 双腕raw | 1,096 | 264.617 | 94.678 | 241.44 → 86.39 | 64.2% |
| 双腕MCAP・実記録10秒抜粋 | 200 | 48.151 | 18.577 | 240.76 → 92.89 | 61.4% |

単腕rawは既存の `/tmp/project-minimal-single-mcap-extract` の実データ503行です。収録rawそのものの450行は欠損があるため、新仕様では正常比較に使用できません。

双腕rawは元の `20261015_164839` 全1,096行。双腕MCAP正常比較は同じ実記録からlog_timeで2〜12秒を抜粋した200行です。メッセージpayload・header・画像を変更せずコピーし、架空データは生成していません。元MCAP全体は終端等に許容差超過を含むため、新仕様では拒否します。

### 工程別の時間（秒）

| 経路・前後 | MCAP | 同期 | 画像decode | 書き出し/encode | 出力検証 | raw読込/その他 | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 単腕raw（既存MCAP抽出済み）・前 | 0.000 | 0.000 | 9.331 | 71.342 | 1.425 | 0.259 | 82.358 |
| 単腕raw（既存MCAP抽出済み）・後 | 0.000 | 0.000 | 6.614 | 11.174 | 1.402 | 0.059 | 19.249 |
| 単腕MCAP・前 | 0.185 | 0.879 | 10.808 | 71.007 | 1.404 | 0.247 | 84.530 |
| 単腕MCAP・後 | 0.123 | 0.000 | 6.799 | 11.521 | 1.414 | 0.007 | 19.864 |
| 双腕raw・前 | 1.738 | 0.047 | 38.110 | 47.943 | 167.123 | 9.655 | 264.617 |
| 双腕raw・後 | 0.000 | 0.000 | 37.568 | 51.244 | 5.667 | 0.199 | 94.678 |
| 双腕MCAP・実記録10秒抜粋・前 | 0.165 | 0.483 | 7.633 | 9.350 | 30.510 | 0.010 | 48.151 |
| 双腕MCAP・実記録10秒抜粋・後 | 0.120 | 0.039 | 7.962 | 9.151 | 1.298 | 0.007 | 18.577 |

工程は計測ラッパーの関数境界に基づく排他的な分類です。画像decodeにはLeRobotの画像統計生成中の画像読み込みも含み、出力検証中のdecodeは検証へ計上します。旧単腕の同期欄には中間raw/JPEG生成、旧双腕rawのMCAP欄には復元関数周辺の処理も含むため、純粋なディスクI/Oだけの時間ではありません。新pipeline自身の工程時間も各JSONの `pipeline_timings` に残しています。

単腕は中間JPEG生成の廃止と一時PNGの無圧縮化が効きました。双腕rawは全フレーム検証が167.123→5.667秒となり、主な改善要因です。一方、双腕rawの書き出し/encode自体は47.943→51.244秒で、今回の測定では短縮していません。

計測器は `scripts/benchmark_conversion.py`、実記録抜粋は `scripts/prepare_conversion_excerpt.py`。変更前を再測定する際は `git archive ec58a7a conversion configs` を別の一時ディレクトリに展開し、その場所を `--source-root` に指定します。変更後は `--source-root .` です。計測用旧ソースは共有コードとして常駐させません。

## 6. 実データと回帰テスト

- 全回帰テスト：39件成功。同期境界、欠損・NaN・型・名前・画像破損・画像shape変更、上書き、失敗時の非公開、8学習設定、正規化、Orbaxの小規模fixture等。
- 実データ4経路：変換・LeRobot reader読み込み成功。20FPS、7/14次元、2/3カメラを確認。
- 前後比較：全行のState・Action・task index・episode/frame indexが一致。task文字列・episode数も一致。時刻は1e-4秒以内。
- カメラ選択：単腕MCAPは旧中間JPEGを全件再現して参照一致、双腕MCAPは全件の元圧縮capture一致、双腕rawは全件の選択元画像一致。単腕抽出済みrawは同じ連番画像を利用。
- 動画は先頭・中央・末尾を比較。単腕MCAPだけ、中間JPEG再圧縮廃止による小さな画素差があり、32×32平均絶対誤差の最大は0.584/255。ほかの比較経路は標本画像で差0。
- 全4経路×π0/π0.5：変更していない学習コードで実バッチを確認。単腕action `(1,20,32)`、双腕 `(1,50,32)`、state `(1,32)`、画像224×224、関節delta化・絶対Gripper・逆変換・paddingを確認。
- 実データのバッチは正規化前。統計計算と正規化後の往復は別fixtureの回帰テストで確認。モデルの重み読み込み・学習・収束試験は実施していません。
- 公開 `sh scripts/convert.sh ... --validate-only`：正常4ケース成功、欠損raw2ケースと許容差超過MCAP1ケースは期待どおり非ゼロ終了。

証跡は `logs/conversion_simplification/tests-all.log`、`comparison.json`、`cli-input-checks.json`、各 `*.timing.json` です。動画の標本比較や学習互換確認は検証用スクリプトだけに置き、標準変換の全画像読み直しは復活させていません。

## 7. 従来と変わった動作・制限

- 欠損rawを名目20Hzへ詰めたり、MCAPで復元したり、episodeを分割したりしません。
- 単腕の元 `20260825_224317` はrow 1後の150ms間隔で拒否。双腕 `20261015_165042` はrow 837後の150ms間隔で拒否します。
- 双腕 `20261015_164839` のMCAP全体は、旧処理では1,099行中3行を無効flag付きで出力できました。新処理ではfrontのrow 1097で134.553msの誤差を検出して拒否します。勝手に終端を切り詰めません。
- rawが不完全でも元MCAPのcaptureが十分なら、MCAPを明示入力にして別datasetへ変換できます。ただしMCAP自身も100ms/30ms条件を満たす必要があります。
- 旧単腕rawにcapture時刻がない場合は、記録済みの同期済み対応を利用します。20Hz列と画像・配列は検査できますが、存在しない元headerの誤差までは再検証できません。
- `--task` があれば明示的に上書きし、なければ記録taskを保持。task欠落時の汎用文言による暗黙補完はありません。
- 旧個別CLIオプションの透過転送はやめ、未対応引数はエラーにします。維持する公開入口は `--robot`・`--dataset`・`--input`・`--output`・`--task`・`--mcap-schema`・`--validate-only` です。
- 詳細provenance・元フレーム時刻列は生成しません。完了状態・入力元・task・フレーム数・shape・工程時間の簡単な記録を残します。

## 8. README・Docker・共有

READMEに単腕/双腕共通の実行例、MCAP/raw配置、固定同期条件、欠損拒否、上書き防止、partial、task、`--validate-only`、valid列の互換理由と制限を記載しました。`conversion/README.md` は各共通モジュールの役割を説明します。旧 `VALIDATION.md` は履歴であることを冒頭に明記し、ソース由来の記録も更新しました。

```sh
sh scripts/convert.sh --robot single --dataset dataset_name
sh scripts/convert.sh --robot bimanual --dataset dataset_name
```

Dockerfileのビルド時import確認先だけを旧converterから新pipelineへ変更しました。Dockerのビルドや既存コンテナの再作成は不要で、実施していません。bind mount側のコードを利用できます。既存 `piper-pi0-sim:111` と共有用コンテナは稼働を維持しました。

共有パッケージには本報告を追加し、dry-runで旧変換コードが含まれないことを確認しました。既存の共有archive自体は上書きしていません。
