# LARRY → Neural SDE → 多項式 SINDy 実験報告

2026-10-07。`1007_instruction.md` の Step 1–7 を順に実行し、負の結果も保存した。
**局所的な teacher 関数の近似は成立したが、安定した symbolic SDE と分岐挙動の保存には至らなかった。**
指示書の実験・成果物は完了しているが、成功判定 M3/M4 は未達である。

## 条件とデータ

Python 3.11.17、PyTorch 2.6.0+cu124、CUDA 12.4、scdiffeq 1.1.4、
neural-diffeqs 0.4.1、torchsde 0.2.6、PySINDy 2.1.0。GPU は RTX 2070 (GPU 0)。
全依存バージョンは [lock](../requirements-lock.txt)、[環境記録](../outputs/logs/environment.txt)。
データは公式 `sdq.datasets.larry(data_dir="data", variant=None)` で取得した。
原本 130,887 cells × 2,492 genes、Quickstart の fate clone・class 条件を適用した
subset は 9,350 cells × 2,492 genes、状態は upstream PCA の50成分。
観測時点は day 2/4/6。[取得元と checksum](../outputs/logs/data_provenance.json)、
[データ inventory](../outputs/logs/dataset_inventory.json) を保存した。

教師学習・split の seed は0。symbolic fit は clone を分離した train 6,137 / validation 1,451 /
test 1,762 cells。変換・状態領域・STLSQ の RMS スケールは training のみで推定した。
validation で次数・閾値・ridge を選び、固定後に test を評価した。
ただし教師の公式 cell-level 学習/検証には292 clonesの重複があり、symbolic test の細胞も
教師学習に多く含まれる。upstream PCA も独立な preprocessing ではない。
**ここでの held-out は symbolic fit に対するもので、独立な生物学的検証ではない。**
[50-PC audit](../outputs/logs/distillation_split_audit.json)、
[5-PC audit](../outputs/logs/pc5_distillation_split_audit.json)。

## Neural SDE と tensor の意味

公式条件を基に、2 epoch smoke の後に新しい seed-0 モデルを1500 epochs学習した。
別途5-PC teacherも同様に最初から学習した。50-PC teacherの射影ではない。

| Teacher | 学習秒 | Peak CUDA bytes | 最終 train loss | 最終 validation loss |
|---|---:|---:|---:|---:|
| 50 PCs | 1784.07 | 2422240256 | 56.0603 | 87.7919 |
| 5 PCs | 1770.34 | 2389755904 | 19.3070 | 22.2976 |

5-PC の損失と50-PC の損失は状態空間が違い、その大小だけで優劣を主張しない。
5 PCs は、選択した training subset の50-PC状態分散の59.57%を保持する。
これは全遺伝子の explained variance ではない。
5-PCへの変更は50-PC validationの不成立を受けて宣言し、最終50-PC test要約を見る前に決定した。
同じcohort上の探索的追加実験である。

実測されたクラスは LightningSDE_FixedPotential_RegularizedVelocityRatio / PotentialSDE。
Itô・general noise・Brownian dimension 1、時間引数を使わない自律系。
scalar potential μ の勾配が drift であり、μの出力は (batch,1) に対して
f は (batch,d)、G は (batch,d,1)、D=GGᵀ は (batch,d,d)。
各PCの独立なBrownian motionではない。public drift/diffusion と直接評価が一致した。
obs の scalar diffusion は G のL2 normであり、GやD自体とは区別する。
[両モデルの意味・shape監査](../outputs/logs/model_semantics.md)。
教師 simulation は有限で、同じseedによる再現を確認した。
公式UMAP図との完全一致や、真の生物学的SDEの同定は主張しない。

## 多項式 distillation

ユーザー指定に従い、定数・べき・交差項のみ。標準化状態 z_i=(PC_i−training mean_i)/training std_i
を用いた degree 1 → 2 の PolynomialLibrary / STLSQ。非多項式関数は含めない。
観測training状態とteacher simulation状態から f/G を直接評価し、軌道の微分推定を代用しなかった。
閾値はtraining RMSで正規化した係数の無次元値。最終非正則化refitは使わずridgeを保持した。
5-PCでdegree2が基準を満たしたためdegree3は実行不要だった。

| Case / field | Library terms | Active coefficients | α / threshold | Validation NRMSE | Test NRMSE | Bootstrap平均選択率 |
|---|---:|---:|---|---:|---:|---:|
| 50-PC drift | 1326 | 2815 | 100 / .05 | .5544 | .6585 | .7311 |
| 5-PC drift | 21 | 51 | 100 / .1 | .4285 | .4710 | .8863 |
| 5-PC G | 21 | 61 | 100 / .1 | .3289 | .3583 | .9541 |

NRMSE=RMSE / RMS teacher target。validationはtraining normの99.5%quantile領域内、
test欄は全query点であり、in-domainのみの値とは区別する。
50-PCの全領域validation NRMSEは9.0655であり、局所値でも事前基準 .5 を満たさなかった。
したがって50-PC diffusionはゲートに従って延期した。
初期unbiased degree2のvalidation NRMSE 56.96という失敗も保存している。

5-PCの局所領域半径は25.4603、validation coverage 99.42%、test 99.84%。
clone単位 bootstrap 5回と閾値±10%を実行した。
5-PC drift の閾値変化時 support Jaccard .9444/.7885、G は .9839/.9032。
主要項には再現性があるが、5 bootstrapのみで厳密な信頼区間を保証しない。
係数は各成分のactive係数数であり、51個/61個の異なる基底関数という意味ではない。

完全な式とスケール変換:
[drift式](../outputs/sindy_pc5/drift/equations.md)、[G式](../outputs/sindy_pc5/diffusion/equations.md)。
[drift結果](../outputs/sindy_pc5/drift/results.json)、[G結果](../outputs/sindy_pc5/diffusion/results.json)。
小さい model.json に全係数・powers・mean/scale を保存し、NPZがなくても復元できる。
Gを直接近似してD=GGᵀを構成するためPSDを保つが、安定性を保証するものではない。
実際の観測状態1,169点でNumPy/Torch評価が一致し、float32 driftの最大差は2.26e−6だった。
[評価実装監査](../outputs/logs/pc5_symbolic_evaluator_audit.json)。

## Simulation と観測分布: 負の結果

固定した同じday2観測細胞32個 × 4replicates × seeds 202/203/204、合計384軌道。
Itô Euler、dt=.05 day、day2/4/6を比較。モデルに状態clippingは加えていない。
非有限値または状態normがtraining領域半径の10倍を超えた軌道を、観測評価時点までの
累積の数値的失敗として数えた。生物学的細胞死ではない。

**day6でsymbolicは225/384=58.59%が失敗、teacherは0/384。**
有限な159軌道の散布図は偏った診断として明示し、fate集計には失敗した225軌道の質量も残した。
全体分布の距離・momentが計算不能な場合はnull/空欄で保存し、0や成功例だけの結果に置き換えない。
dt=.025の追加確認でもsymbolic非有限値が発生した。
これは試した数値積分での不安定性であり、連続時間SDEの数学的爆発の証明ではない。
Brownian entropyの一致だけで刻み変更時のpathwise convergenceも主張しない。

| Day6 composition | Observed data | Neural SDE teacher | Symbolic SDE |
|---|---:|---:|---:|
| Monocyte | .3911 | .5182 | .1771 |
| Neutrophil | .3616 | .4661 | .0938 |
| Undifferentiated | .2473 | .0156 | .1432 |
| Numerical failure | 0 | 0 | .5859 |

teacherにも観測組成とのずれがあり、teacher一致だけでは生物学的妥当性を保証しない。
KNN readoutはsymbolic training観測細胞のみでfitし、test分類精度 .9455。
記述的なfate readoutであり、独立なfate prediction benchmarkではない。
失敗質量を含むteacher/symbolic total variationは .7135。
training標準化座標上の64方向sliced Wasserstein distanceは、day4で
teacher↔observed .2527、symbolic↔observed 1.0696、symbolic↔teacher 1.0680。
day6 teacher↔observed .3765、symbolic全体は非有限値のため評価不能。

[全結果](../outputs/validation_pc5/results.json)、[安定性](../outputs/validation_pc5/stability.csv)、
[距離](../outputs/validation_pc5/distribution_distances.csv)、[fate](../outputs/validation_pc5/fate_composition.csv)。
[endpoint図](../outputs/figures/pc5_endpoint_comparison.png)、[fate図](../outputs/figures/pc5_fate_comparison.png)。
独立にfitした多項式driftはteacherのpotential勾配構造を強制していない。
局所誤差が小さくPSDが保たれていても、積分による領域外への逸脱・分布崩壊を防げなかった。
検証結果を見て閾値・次数・初期値を再調整していない。

## Raw direct baseline の意味

LARRYは同一細胞の連続時系列ではない。そのためcloneの2日間centroid変位を
直接多項式回帰する限定的drift proxyを作った。train310 / validation75 / test90 pairs。
同じclonecentroidを初期値として、teacher/symbolicは16replicateの2日間積分平均で比較した。
clonecentroidの変位には分岐・増殖・サンプリングも含まれる。
このpseudo-transitionからintrinsic diffusionを識別できるとして、stochastic Gを捏造しなかった。

| 2-day centroid displacement test NRMSE | 50 PCs | 5 PCs |
|---|---:|---:|
| Raw polynomial proxy | .8428 | .8306 |
| Neural SDE teacher | .9673 | 1.0876 |
| Symbolic SDE | 未実行（drift gate失敗） | 1.1970 |

この限定指標ではraw proxyが良い。公正なstochastic-SINDy比較やteacherの優位性の証拠とは扱わない。
2日間centroid開始の積分は有限だったが、前節の4日間single-cell開始とは条件が異なる。
[仮定](direct_baseline_assumptions.md)、[5-PC結果](../outputs/direct_baseline_pc5/results.json)。

## 完了範囲と再現

| Milestone | 判定 |
|---|---|
| M0 Data | 完了 |
| M1 Neural SDE | 学習・直接評価・simulationの実行を確認 |
| M2 Drift SINDy | 50-PCは不十分、探索的5-PCの局所teacher近似は事前基準を満たす |
| M3 Symbolic SDE | Gの式は得たが積分不安定、未達 |
| M4 Biological demo | 主要分布・fateを保持せず、未達 |

説明用Notebook 00–05を実行し、8 tests passed。成果物はoutputsに保存した。
checkpoint・生データ・simulation配列は大容量のためGit外、取得・生成コマンドとchecksumを記録。
50-PC checkpoint SHA256:
`a487bcfa6e89c62ee2cb4a8bcd0af3a533b651b71efc095f1ea6d342df5658d2`。
5-PC checkpoint SHA256:
`59d751c5241c85adef62ccc96428bf6eb5d848b76d41326355c9c233f659b0ce`。
再学習はCUDA環境に依存し、異なるhardwareで同一checkpoint hashになるとは保証しない。

環境・全コマンドは [Notebook README](../notebooks/README.md)。基本順序は
`src.data → src.baseline → src.prepare → src.run_distillation → src.validate → src.direct_baseline`。
5-PC runは `configs/experiment_pc5.json` と各CLIのpc5出力指定を使う。
50-PCは `configs/experiment.json`。全式とJSONモデルを読み込むだけならcheckpointは不要。

中心仮説は今回の条件では支持されなかった。
次の研究では独立な教師学習・preprocessing splitと、training/validationのみで選ぶ
安定性またはpotential構造を保つ多項式表現を設計し、**新しい未使用の評価集合**で比較する。
今回のtest結果を使った救済的tuningはこの実験に含めない。
