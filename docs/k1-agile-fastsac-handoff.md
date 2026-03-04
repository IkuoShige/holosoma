# K1 Agile FastSAC 引き継ぎ資料

## 経緯

K1 agile FastSAC で速度指令 0 の時に関節が発振（プルプル）する問題の修正に取り組んだ。

## 実施した変更 (branch: `feat/k1-fast`)

### 1. L1 action rate penalty の追加
- **ファイル**: `src/holosoma/holosoma/managers/reward/terms/locomotion.py`
- L2 penalty は 0.01rad の微小振動に対して gradient が消失する (penalty ≈ 0.0001)
- L1 は constant gradient を持つため、微小振動を効果的に抑制
- **weight = -0.3** (`k1_22dof_agile_loco_fast_sac` reward config)

### 2. FastSAC entropy ratio の調整
- **ファイル**: `src/holosoma/holosoma/config_values/loco/k1/experiment.py`
- `target_entropy_ratio`: 0.25 → **0.18**
- 0.12 は過度に決定的で学習不安定、0.18 がバランス良好

### 3. ゼロ速度学習サンプルの増加
- **ファイル**: `src/holosoma/holosoma/config_values/loco/k1/command.py`
- `stand_prob`: 0.1 → **0.2**
- ゼロ速度での学習機会を 2 倍に

### チューニング経緯

| 試行 | L1 weight | entropy_ratio | 結果 |
|------|-----------|--------------|------|
| v1   | -0.8      | 0.12         | tracking -14%/-21% 劣化、episode length 低下 |
| **v2** | **-0.3** | **0.18** | **tracking 維持、episode length 改善** |

## 学習メトリクス比較 (wandb summary, 100k iter)

| メトリクス | ベースライン (02/27) | v2 (03/02) | 変化 |
|---|---|---|---|
| tracking_lin_vel (raw) | 23.6 | 23.9 | +1.4% |
| tracking_ang_vel (raw) | 18.8 | 18.6 | -1% |
| action_rate L2 (raw) | 108.0 | 97.4 | -10% (改善) |
| action_rate L1 (raw) | — | 232.6 | (新規) |
| episode length | 878 | 893 | +2% |
| orientation (raw) | 0.308 | 0.193 | -37% (改善) |
| termination (raw) | 0.011 | 0.015 | +36% (微増) |

**学習指標上は tracking 性能を維持しつつ action rate を改善**

## Eval 結果 (eval_tracking.py, IsaacSim)

チェックポイント: `logs/hv-k1-manager/20260302_233930-.../model_0100000.pt`

```
シナリオ                   指令       実測     追従率   振動std  転倒
停止 (stand)             vx=+0.0   vx=+0.001          0.0055   N
前進 (forward)           vx=+1.0   vx=+0.887    89%   0.0835   N
後退 (backward)          vx=-0.8   vx=-0.780    97%   0.0353   N
左移動 (strafe L)        vy=+0.5   vy=+0.407    81%   0.1209   N
右移動 (strafe R)        vy=-0.5   vy=-0.517   103%   0.1051   N
直進+旋回 (fwd+turn)     vx=+1.0   vx=+0.896    90%   0.0907   N
                         yw=+0.8   yw=-0.183   -23%            N
高速前進 (fast fwd)      vx=+2.0   vx=+0.565    28%   0.1140   N

停止時振動: vx_std=0.0055, vy_std=0.0102, yaw_std=0.0840
```

## 残存問題

### P1: yaw 追従の破綻
- **症状**: yaw=+0.8 指令に対して -0.183（逆方向に回る）
- **影響**: 直進+旋回の実用性がない
- **仮説**:
  - 学習時の tracking_ang_vel reward (raw=18.6) は高いのに eval で出ない → 学習時は平均的に良いが、特定の速度域で破綻している可能性
  - eval で command を固定し続けている (学習時は 8s でリサンプリング) ため、長時間同じ旋回指令への追従が学習されていない可能性
  - sim-to-sim gap (学習は IsaacSim、eval も IsaacSim だが num_envs=1 vs 4096)
- **確認**: ベースラインでも同じ eval を走らせて比較すべき

### P2: 高速域 (vx=2.0) の追従率が 28%
- **症状**: 2.0 m/s 指令に対して 0.565 m/s しか出ない
- **影響**: agile を名乗る以上、高速域が出ないと意味がない
- **仮説**:
  - 学習時の command range は vx=[-1.5, 2.5] だが、combined_motion_prob=0.3 のケースでのみ 1.4-2.5 の高速域がサンプリングされる → 高速単独の学習が不足
  - penalty_stall_when_commanded (w=-4.0) が効いているはずだが eval では地形の影響？
  - gait frequency の adaptive range (1.2-2.5 Hz) が高速時に正しく動いているか
- **確認**: eval 中の gait_freq を記録して確認

### P3: 停止時の yaw 振動 (yaw_std=0.084)
- **症状**: 並進はほぼ静止だが yaw 方向に 0.084 rad/s の揺れ
- **影響**: カメラブレ、見た目の安定性
- **仮説**:
  - L1 penalty は action 全体に効くが、yaw に対する足位置の感度が低い
  - penalty_head_ang_vel_xy はあるが yaw 方向の penalty がない
- **対策案**: 停止時の yaw angular velocity に特化した penalty 追加

### P4: strafe left の追従率が 81%
- 右は 103% なのに左が 81%。左右非対称。
- symmetry loss (`use_symmetry=True`) が効いているはずだが完全ではない

## 使い方

### 学習
```bash
source scripts/source_isaacgym_setup.sh  # or source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py exp:k1-22dof-agile-fast-sac simulator:isaacgym logger:wandb
```

### Eval (速度追従テスト)
```bash
# Xvfb が必要 (display なし環境)
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
source scripts/source_isaacsim_setup.sh
python scripts/eval_tracking.py \
    --checkpoint=logs/hv-k1-manager/<run_dir>/model_0100000.pt \
    --eval-overrides.headless True --logger.video.enabled False
```

### Eval (動画付き)
```bash
python scripts/eval_tracking.py \
    --checkpoint=<path> \
    --eval-overrides.headless True --logger.headless-recording True
```

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/holosoma/holosoma/managers/reward/terms/locomotion.py` | `penalty_action_rate_l1` 関数追加 |
| `src/holosoma/holosoma/config_values/loco/k1/reward.py` | L1 term を `k1_22dof_agile_loco_fast_sac` に追加 (w=-0.3) |
| `src/holosoma/holosoma/config_values/loco/k1/experiment.py` | `target_entropy_ratio` 0.25→0.18 |
| `src/holosoma/holosoma/config_values/loco/k1/command.py` | `stand_prob` 0.1→0.2 |
| `scripts/eval_tracking.py` | 速度追従評価スクリプト (新規) |

## 次のアクション

1. **ベースライン比較**: 変更前のチェックポイント (`20260227_223327-...`) で `eval_tracking.py` を走らせ、P1-P4 が元からあったか確認
2. **yaw 追従の調査**: eval 中の tracking_ang_vel reward を step ごとに記録し、どこで破綻しているか特定
3. **高速域の改善**: `combined_motion_prob` の増加、または高速 vx 単独のサンプリング追加
4. **yaw 静止振動**: 停止時 yaw penalty の追加検討
