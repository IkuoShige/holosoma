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
