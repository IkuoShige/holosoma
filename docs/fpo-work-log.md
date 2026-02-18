# FPO (Flow Policy Optimization) 作業ログ

## 実施日: 2025-02-15
## ブランチ: feat/k1

---

## 作業概要

HolosomaコードベースにFPO（Flow Policy Optimization, arXiv:2507.21053）を新規アルゴリズムとして追加。
PPOのフレームワーク上で条件付きフローマッチング（CFM）ベースのポリシーを学習するon-policyアルゴリズム。

## 新規作成ファイル

| ファイル | 内容 |
|---------|------|
| `src/holosoma/holosoma/agents/fpo/fpo_agent.py` | FPOAgent（PPO継承、FPO比率計算、MC sample保存、ONNX export、推論wrapper） |
| `src/holosoma/holosoma/agents/modules/flow_policy.py` | FlowPolicy（TimestepEmbedder, AdaLNBlock, VelocityFieldMLP, ODE積分） |

## 変更した既存ファイル

| ファイル | 変更内容 |
|---------|---------|
| `config_types/algo.py` | `FPOConfig`, `FPOModuleDictConfig`, `FPOAlgoConfig` 追加。Union型拡張 |
| `config_values/algo.py` | `fpo` デフォルトインスタンス追加、DEFAULTS登録 |
| `agents/modules/module_utils.py` | `setup_flow_policy_module()` 追加 |
| `config_values/experiment.py` | g1/t1/k1のFPOプリセットをDEFAULTSに登録 |
| `config_values/loco/g1/experiment.py` | `g1_29dof_fpo` 追加 |
| `config_values/loco/t1/experiment.py` | `t1_29dof_fpo` 追加 |
| `config_values/loco/k1/experiment.py` | `k1_22dof_fpo` 追加 |

## 検証済み項目

- ruff lint / format 全ファイルパス
- Config import & experiment registration 動作確認
- FlowPolicy 補間テスト（t=0→action, t=1→eps, 線形性）
- CFM loss 形状テスト（[B,K,1]）
- ODE積分テスト（act/act_inference → [B,A]）
- FPO ratio計算テスト（per_sample, clamping）
- ONNX export（CPU/CUDA roundtrip、constant folding問題解決済み）

## 未検証項目

- checkpoint保存・復元
- PPOベースラインとの学習曲線比較
- 推論ステップ数K削減時の品質劣化評価

---

## バグ修正履歴

### 1. sinusoidal_embedding の broadcasting エラー
- **症状**: `compute_flow_loss` で `RuntimeError: size mismatch` (t=[B*K] vs freq=[half_dim])
- **原因**: `t.squeeze(-1)` で [B*K,1] → [B*K] にした後、freq [half_dim] との乗算で broadcasting 失敗
- **修正**: `t.unsqueeze(-1)` で [B,1] を保証し、`freq.unsqueeze(0)` で [1, half_dim] にして broadcasting

### 2. ONNX export デバイス不整合 (1回目)
- **症状**: `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`
- **原因**: `TimestepEmbedder.sinusoidal_embedding` 内の `torch.arange` がONNX constant folding時にCPU定数になる
- **修正**:
  - `torch.arange` を `__init__` で `register_buffer("freq", ...)` に変更（`.to(device)` で自動移動）
  - `torch.full(..., device=)` → `obs.new_full(...)` に変更（入力テンソルからdevice継承）
  - ONNX wrapper内の `torch.zeros` → `actor_obs.new_zeros` に変更

### 3. ONNX export デバイス不整合 (2回目)
- **症状**: 同じ `RuntimeError`、修正1適用後も発生
- **原因**: PPOの `export()` が `_OnnxMotionPolicyExporter` → `_extract_actor_model_and_input_dim` 経由で FlowPolicyONNXWrapper を**分解**し、内部の velocity_field だけ取り出す。ODE積分ループが消失し、velocity_field が直接呼ばれてデバイス不整合発生
- **修正**:
  - `FPOAgent.export()` をオーバーライドし、`_OnnxMotionPolicyExporter` を通さず `export_policy_as_onnx` を直接呼び出す
  - export時に wrapper 全体を CPU に移動してから export → 終了後 device に戻す

---

## FPOで歩行制御policyを学習するコマンド例

### 環境セットアップ

```bash
# IsaacGym環境の場合
source scripts/source_isaacgym_setup.sh

# IsaacSim環境の場合
source scripts/source_isaacsim_setup.sh

# MuJoCo/MJWarp環境の場合
source scripts/source_mujoco_setup.sh
```

### G1ロボット（29DoF）

```bash
# IsaacGym
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-fpo simulator:isaacgym logger:wandb \
  --training.seed 1

# MJWarp
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-fpo simulator:mjwarp logger:wandb \
  --training.seed 1
```

### T1ロボット（29DoF）

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:t1-29dof-fpo simulator:isaacgym logger:wandb \
  --training.seed 1
```

### K1ロボット（22DoF）

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:k1-22dof-fpo simulator:isaacgym logger:wandb \
  --training.seed 1
```

### パラメータカスタマイズ例

```bash
# ODE積分ステップ数を変更（デフォルト: 10）
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-fpo simulator:isaacgym \
  --algo.config.num_flow_steps 8

# MCサンプル数を変更（デフォルト: 4）
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-fpo simulator:isaacgym \
  --algo.config.num_mc_samples 8

# クリップパラメータを変更（デフォルト: 0.05）
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-fpo simulator:isaacgym \
  --algo.config.clip_param 0.1

# legacy_avgモードで実行（デフォルト: per_sample）
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-fpo simulator:isaacgym \
  --algo.config.ratio_mode legacy_avg

# adaLNを無効化（timestep embeddingをconcatで注入）
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-fpo simulator:isaacgym \
  --algo.config.use_ada_ln False
```

### 評価

```bash
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=/path/to/model_XXXXX.pt \
  --training.headless=False
```

---

## FPO++ 再現実装の発散修正 (2026-02-18, feat/fpo)

### 問題

`g1_29dof_fpo_pp_repro` (FPO++, arxiv:2602.02481) の学習で cfm_loss が ~700K に発散。ロボットの動き自体は改善傾向にあるが、velocity field が不安定で学習が収束しない。

### 根本原因

1. **num_steps_per_env=96, num_mini_batches=16** — 論文は 24, 4。1イテレーションあたり 512 回の gradient update（論文は128回）で old_loss が陳腐化
2. **Stage 1 CFM loss clamping が ratio 計算に未適用** — `cfm_loss_clip` のクランプが診断メトリクス計算（no_grad ブロック内）にしか存在せず、実際の `_compute_fpo_ratio` には反映されていなかった。論文は「(i) clamping CFM losses before taking differences and (ii) then clamping the difference before exponentiation」と明記

### 修正内容

| ファイル | 変更 |
|---------|------|
| `config_values/loco/g1/experiment.py` | `num_steps_per_env`: 96→24, `num_mini_batches`: 16→4, `cfm_loss_clip=5000.0` 追加 |
| `agents/fpo/fpo_agent.py` | `_compute_fpo_ratio` に stage 1 clamping 追加（old_loss/new_loss を cfm_loss_clip でクランプしてから diff 計算） |

### 検証メトリクス

- `cfm_loss`: 発散しないか（初期 ~300-500 が安定推移すること）
- `fpo_clamp_stage1_rate`: stage 1 clamping 発火率
- `fpo_clamp_stage2_rate`: 0.5 以下に低下するか
- `fpo_sign_agree`: 0.5 より上昇するか
- `fpo_actor_grad_norm`: max_grad_norm (0.5) から徐々に下がるか

---

## 主要な設計判断

| 項目 | 決定 | 理由 |
|------|------|------|
| 補間スケジュール | `x_t = t*eps + (1-t)*action` | 参照実装に準拠。t=0:clean, t=1:noise |
| 比率計算 | per_sample (FPO++) デフォルト | 各MCサンプルで個別に比率計算→平均。安定性向上 |
| Critic | PPOCriticをそのまま再利用 | 値関数はFPOでも同一 |
| エントロピー正則化 | なし (coef=0.0) | フローモデルにはエントロピーの閉形式がない |
| 適応学習率 | なし | FPOではKL推定が未確立 |
| MC samples保存 | RolloutStorageに[K,*]形状で追加 | x_tは再計算（メモリ節約） |
| ONNX export | `FPOAgent.export()`オーバーライド | PPOの`_OnnxMotionPolicyExporter`がODE loopを分解してしまうため回避 |
| ONNX初期ノイズ | zeros（決定論的） | ONNX exportとデプロイ時の再現性のため |
| テンソル生成 | `new_zeros`/`new_full`/`register_buffer` | ONNX constant folding時のデバイス不整合防止 |

---

## FPO++ G1 29DOF 再現実験: 総合報告 (2026-02-18)

### 1. 結論サマリ

**from-scratch FPO++ は G1 29DOF locomotion で現時点 non-viable 寄り。**

- 7回の実験を通じ、安定性と学習信号のトレードオフを解消できなかった
- 崩壊しない設定 (action_bound=1.5) では cov_adv_logr ≈ 0.006 で学習信号が弱すぎ、歩行を獲得できず
- 学習信号を強める設定 (action_bound≥2.0) では flow model が崩壊
- **次のステップ**: PPO warm-start → FPO fine-tune の切り分け実験を1本実施し、from-scratch 固有の問題か実装レベルの問題かを判別

### 2. 再現目的と判定基準

- **目的**: FPO++ (arxiv:2602.02481) の G1 29DOF locomotion での再現
- **成功条件**: PPO 同等の歩行獲得 (tracking reward 上昇、episode length 延伸)
- **打ち切り条件**: ハイパラ探索が3段階以上の分岐を経ても改善が見られない場合

### 3. 実験固定条件

| 項目 | 値 |
|------|-----|
| 環境 | `LeggedRobotLocomotionManager` |
| ロボット | `g1_29dof` |
| 観測 | `g1_29dof_loco_single_wolinvel` |
| 行動 | `g1_29dof_joint_pos` (action_scale=0.25) |
| 報酬 | `g1_29dof_loco` |
| シミュレータ | IsaacSim |
| Flow steps | 64 |
| MC samples | 16 |
| ratio_mode | per_sample |
| trust_region | aspo |
| flow_param_mode | velocity |

### 4. 実験タイムライン

| # | 変更点 | 主要観測 (iter) | 判定 |
|---|--------|----------------|------|
| 1 | 初期 (epochs=32, clip=0.5, steps=24, mini=4) | cfm_loss ~700K 発散 | 失敗: stage1 clamping なし |
| 2 | + stage1 clamping 実装 | cfm_loss 安定、delta_loss_std=651, stage2_rate=50% | 失敗: old_loss 陳腐化 |
| 3 | epochs=12, ratio_log_clip=1.0, cfm_loss_clip=10000 | delta_loss_std=124→改善、cov=0.026 (iter50) | 部分成功B: stage2_rate=50% |
| 4 | epochs=8 | delta_loss_std=21.7、cov=0.001 (iter63) | 失敗: 信号消失 |
| 5 | cfm_loss_reduction=sum→mean | cov=0.009(iter73)→-0.018(iter157) | 失敗: 一時回復後崩壊 |
| 6 | action_bound warmup 0.5→3.0 | warmup中: cov=0.057(iter50, 過去最高)。3.0到達後: 崩壊 | Gate A pass, Gate B fail |
| 7a | action_bound warmup 0.5→2.0 | warmup中: cov=0.038(iter50)。2.0到達後: 崩壊 | Gate A pass, Gate B fail |
| 7b | action_bound=1.5 固定 | cov=0.005-0.007安定。reward +0.77(iter159)。iter234でcfm_loss爆発 | **初の安定 run だが歩行未獲得** |

### 5. 因果推論

**確立された因果関係** (同一 run 内での操作で確認):
- `action_bound ≥ 2.0` → raw_logr_std 爆発 → stage2 clamp 増加 → cov_adv_logr 崩壊
- `action_bound ≤ 1.5` → raw_logr_std 安定 → stage2_rate < 12% → cov 安定 (ただし低位)
- `cfm_loss_reduction=sum` → cfm_loss スケール過大 → log_ratio 分散増大
- `cfm_loss_reduction=mean` → cfm_loss スケール正常化

**相関だが因果未確定**:
- cov_adv_logr ≈ 0.006 が学習の限界なのか、他の要因で抑制されているのか
- flow model の iter 200+ での崩壊が action_bound とは独立した固有の不安定性か

### 6. 失敗モード辞書

| モード | 定義 | 主要メトリクス | 対処 |
|--------|------|---------------|------|
| CFM 発散 | cfm_loss が指数的に増大 | cfm_loss > 1000, cfm_loss_ratio > 1.5 | stage1 clamping, cfm_loss_clip |
| Ratio 飽和 | log_ratio が clip 境界に張り付き | stage2_rate > 40%, ratio_p10=e^{-clip} | action_bound 縮小, reduction=mean |
| 信号消失 | advantage と ratio の相関喪失 | cov_adv_logr < 0.005, sign_agree ≈ 0.50 | action_bound warmup, epochs 削減 |
| Flow fit 崩壊 | flow model の表現力劣化 | cfm_loss 急増 (>100), raw_logr_mean << -10 | 未解決 |
| Local minimum | 倒れないが歩かない | reward 横ばい、episode_length 一定 | 未解決 (構造的問題の疑い) |

### 7. 現時点の総合判断

**from-scratch FPO++ の継続を停止する。**

根拠:
- 安定動作する設定 (action_bound=1.5) で cov_adv_logr ≈ 0.006 は、locomotion 学習に不十分
- PPO は同一環境で数千 iter で歩行を獲得する
- 7回の実験で一貫して ratio ベース学習信号の弱さが確認された

**未確定事項**:
- 実装にバグがある可能性はゼロではない（論文は Humanoid で動作を主張）
- PPO warm-start → FPO fine-tune で切り分け可能

### 8. 次実験計画: PPO warm-start → FPO fine-tune

**仮説**: FPO++ の from-scratch 学習が弱いのであって、合理的な初期 policy からの fine-tune は機能する

**手順**:
1. PPO で G1 29DOF を iter 5000 まで学習（歩行獲得を確認）
2. PPO の actor weights を FPO の flow model に移植（要設計）
3. FPO fine-tune を iter 500 実行
4. 報酬と歩行品質を PPO 継続と比較

**評価メトリクス**: tracking reward, episode_length, cov_adv_logr

**Gate**:
- iter 100: tracking_lin_vel が PPO baseline の 50% 以上を維持
- 失敗: FPO fine-tune も機能しない → 実装監査へ

### 9. 最終意思決定マトリクス

| warm-start 結果 | 解釈 | 次のアクション |
|-----------------|------|---------------|
| fine-tune 成功 (歩行維持+改善) | from-scratch 学習の初期化問題 | FPO++ を warm-start 前提で運用 |
| fine-tune で性能維持だが改善なし | FPO ratio 学習が PPO に勝る根拠なし | FPO++ 路線保留、PPO に集中 |
| fine-tune で性能劣化 | FPO 実装に問題の疑い | 実装監査 (ratio 計算、flow loss) |

### 10. 再現用アーティファクト

| 項目 | 値 |
|------|-----|
| ブランチ | `feat/fpo` |
| 実験 config | `g1_29dof_fpo_pp_repro` |
| 実行コマンド | `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fpo-pp-repro simulator:isaacsim logger:wandb --training.seed 1` |
| 最終 config 値 | action_bound=1.5, warmup_iters=0, epochs=8, mini_batches=4, steps=24, ratio_log_clip=1.0, cfm_loss_clip=10000, reduction=mean |
| 変更ファイル | `config_types/algo.py`, `config_values/loco/g1/experiment.py`, `agents/fpo/fpo_agent.py` |
