#!/usr/bin/env bash
# ============================================================================
# run_major_revision_audits.sh — Array 大修（ARRAY-D-26-07014）补充实验调度
#
# 对应修改计划 paper/Array/ARRAY_major_revision_plan.md §1–§4：
#   s1a  §1.2 主结果·枕区 2 单元   pooled 置换 B=5000, S=20  → seed_inference_audit.json
#   s1b  §1.2 主结果·全脑 3 单元   pooled 置换 B=2000, S=5   → seed_inference_audit.json（合并）
#   s1c  §1.3 置换单位敏感性       within 被试段内置换（枕区） → seed_inference_audit_within.json
#   s1d  §1.4 全脑 18-cell 家族补齐（5 时间窗 × 3 target，见下方第三轮补跑）
#   s2a  §2 预算匹配·配置 A        pool（总预算 200/fold）    → budget_audit_pool.json
#   s2b  §2 预算匹配·配置 B        per_subject（200/被试）    → budget_audit_per_subject.json
#   s2c  §2 预算匹配·配置 C        max-trials inf（全部 trial）→ budget_audit_all.json
#   s3a  §3.1 第一类错误率校准     pooled，n_rep=1000        → seed_inference_audit_calib.json
#   s3b  §3.1 校准·within 单位                                → seed_inference_audit_calib_within.json
#   posthoc §3.3 功效/MDE + §4.1 Fisher/Stouffer + §4.2 seed 表 + §1.4 BH 家族（离线，秒级）
#
# ---- 第二轮补跑（2026-11-10，针对第一轮核查暴露的 5 个缺口）------------------
#   f1   §1.4 21-cell ROI 家族正式 p 值 + BH
#          pooled 置换 B=5000, S=20，7 ROI × 3 target = 21 cell
#          → seed_inference_audit_roi_family.json
#   f2   §2 全脑 3 cell 的配置 B（per_subject 200/被试）
#          → budget_audit_per_subject_scalp.json
#   f3   §2 全脑 3 cell 的配置 C（全部训练 trial）
#          → budget_audit_all_scalp.json
#   f4   §1.3 全脑 3 cell 的 within 置换敏感性
#          → seed_inference_audit_scalp_within.json
#   f5   §1.4 提高置换次数以突破 p 分辨率下限（枕区 配置 B / C）
#          → budget_audit_per_subject_hib.json / budget_audit_all_hib.json
#   f6   §1.3+§1.4 ROI 家族的 within 置换敏感性
#          → seed_inference_audit_roi_family_within.json
#
# ---- 第三轮补跑（2026-11-17，全脑 18-cell 时间窗家族）------------------------
#   s1d  §1.4 全脑 18-cell 家族补齐：6 时间窗 × 3 target 中，full 窗 3 cell
#          已由 s1b 完成，此处补 N1/P2/P3/LPP/LATE 5 窗 × 3 target = 15 cell
#          pooled 置换 B=2000, S=5（与 s1b 同口径）
#          → seed_inference_audit_scalp_{N1,P2,P3,LPP,LATE}.json（每窗一文件）
#
# 用法（在工作站项目根目录下）：
#   bash code/run_major_revision_audits.sh <stage>
#   stage ∈ smoke | s1a | s1b | s1c | s1d | s2a | s2b | s2c | s3a | s3b | posthoc
#          | f1 | f2 | f3 | f4 | f5 | f6 | all
#
#   推荐在 tmux 中运行（断开 SSH 不影响任务）：
#     tmux new -s array                              # 新建会话
#     bash code/run_major_revision_audits.sh smoke   # 先冒烟（含真实数据 6 被试试跑）
#     bash code/run_major_revision_audits.sh all 2>&1 | tee logs/all.log
#     # Ctrl-b 然后 d 脱离会话；tmux attach -t array 回到会话
#   也可每阶段独立会话（互不影响、便于并行观察）：
#     tmux new -d -s f1 'bash code/run_major_revision_audits.sh f1 2>&1 | tee logs/f1.log'
#
# 可调环境变量（默认值已按修改计划设定）：
#   PY=python  NJOBS=0(自动 cpu-4)  B_OCC=5000  B_ALL=2000  S_OCC=20  S_ALL=5
#   B_BUDGET=2000  N_REP=1000  N_PERM_CAL=2000  N_BLOCKS=2500
#   SIDED=one            正式 p 值方向口径（one=单侧上尾 / two=双侧）
#                       ※ §1.2 要求事先固定，**不得**在看到结果后更改
#   B_ROI_FAMILY=5000    f1 的置换次数（21 cell × B=5000 × S=20 ≈ 8.6 万次解码）
#   S_ROI_FAMILY=20      f1 的 seed 数
#   B_SCALP_BUDGET=1000  f2/f3 全脑配置 B/C 的置换次数
#                       ※ 全脑 per_subject 单次解码实测 ≈347s（枕区 5s），
#                         故默认取 1000 次而非 2000 次；见下方耗时估算
#   S_SCALP_BUDGET=5     f2/f3 全脑配置 B/C 的 seed 数
#   B_HIB=20000          f5 的置换次数（突破 1/(B+1) 分辨率下限）
#
# 耗时估算（工作站实测 24 核 / n_jobs=20；单次解码耗时为实测值）：
#   枕区 9ch  pool        0.26s/次 → B=5000 × S=20 = 10 万次 ≈ 2.6h/cell
#   枕区 9ch  per_subject 5.05s/次 → B=2000 × S=5  = 1 万次 ≈ 14h/cell
#   全脑 64ch pool        4.5s/次  → B=2000 × S=5  = 1 万次 ≈ 12.5h/cell
#   全脑 64ch per_subject 347s/次  → B=1000 × S=5  = 5 千次 ≈ 482h/cell（3 cell 共 ~60 天）
#
#   f1  21 cell：ROI 通道数 6–16ch（枕区 9ch 实测 0.26s/次 → 约 2.6h/cell），
#       按通道数线性外推总解码 ≈ 8.6 万次，估计 12–20h
#   f2  全脑配置 B：3 cell × 5 千次 × 347s ≈ 60 天 —— **超出合理工期，默认不跑**
#   f3  全脑配置 C：同 f2，默认不跑
#   f4  全脑 within：3 cell × 1 万次 × 4.5s ≈ 37.5h（pool 预算，可接受）
#   f5  枕区配置 B/C 提高 B：2 cell × 2 配置 × 2 万次 × 5s ≈ 111h
#   f6  ROI 家族 within：同 f1 规模，约 12–20h
#   s1d 全脑 15 cell（5 窗 × 3 target）：实测单 cell（B=2000,S=5,64ch pool）
#       4980s ≈ 1.38h → 15 cell 串行 ≈ 20.8h
#
#   → f2/f3 默认**不纳入 all**，需显式指定 stage（原因见上）。
#     若确需全脑预算匹配，建议先用 B_BUDGET 小预算（如 100）验证管线，
#     或只跑 rating_taste 一个 target。
#
# 断点续跑：同参数重复执行会自动跳过已完成 cell，中断后直接重跑即可。
# ============================================================================
set -euo pipefail

PY="${PY:-python}"
NJOBS="${NJOBS:-0}"
B_OCC="${B_OCC:-5000}"
B_ALL="${B_ALL:-2000}"
S_OCC="${S_OCC:-20}"
S_ALL="${S_ALL:-5}"
B_BUDGET="${B_BUDGET:-2000}"
N_REP="${N_REP:-1000}"
N_PERM_CAL="${N_PERM_CAL:-2000}"
N_BLOCKS="${N_BLOCKS:-2500}"
SIDED="${SIDED:-one}"
B_ROI_FAMILY="${B_ROI_FAMILY:-5000}"
S_ROI_FAMILY="${S_ROI_FAMILY:-20}"
B_SCALP_BUDGET="${B_SCALP_BUDGET:-1000}"
S_SCALP_BUDGET="${S_SCALP_BUDGET:-5}"
B_HIB="${B_HIB:-20000}"
R=outputs/results

OCC_CELLS="occipital|rating_taste,occipital|rating_willingnessToEat"
ALL_CELLS="all|rating_taste,all|rating_health,all|rating_willingnessToEat"
CALIB_CELL="occipital|rating_taste"

mkdir -p "$R" logs

smoke() {
    # 合成数据 selftest ×4：验证统计逻辑（unit/semantics/inf 三变体）
    $PY code/audit_seed_inference.py --selftest
    $PY code/audit_seed_inference.py --selftest --unit within
    $PY code/audit_seed_inference.py --selftest --semantics per_subject
    $PY code/audit_seed_inference.py --selftest --max-trials inf \
        --cells "$CALIB_CELL"
    # 双侧口径 selftest（§1.2 方向假设）
    $PY code/audit_seed_inference.py --selftest --sided two
    # 真实数据环境冒烟（6 被试 × 3 置换）：验证 Linux 数据路径与依赖全链路
    rm -f "$R/_smoke_linux.json"
    $PY code/audit_seed_inference.py --stage cell --nsub 6 --perm 3 --seeds 3 \
        --cells "$CALIB_CELL" --out "$R/_smoke_linux.json" --n_jobs "$NJOBS"
    echo "[smoke] 全部通过（selftest ×5 + 真实数据 6 被试试跑）"
}

s1a() { $PY code/audit_seed_inference.py --stage cell \
    --perm "$B_OCC" --seeds "$S_OCC" --cells "$OCC_CELLS" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

s1b() { $PY code/audit_seed_inference.py --stage cell \
    --perm "$B_ALL" --seeds "$S_ALL" --cells "$ALL_CELLS" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

s1c() { $PY code/audit_seed_inference.py --stage cell --unit within \
    --perm "$B_BUDGET" --seeds "$S_OCC" --cells "$OCC_CELLS" \
    --out "$R/seed_inference_audit_within.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

s2a() { $PY code/audit_seed_inference.py --stage cell --semantics pool \
    --perm "$B_BUDGET" --seeds "$S_ALL" --cells "$OCC_CELLS" \
    --out "$R/budget_audit_pool.json" --n_jobs "$NJOBS" --sided "$SIDED"; }

s2b() { $PY code/audit_seed_inference.py --stage cell --semantics per_subject \
    --perm "$B_BUDGET" --seeds "$S_ALL" --cells "$OCC_CELLS" \
    --out "$R/budget_audit_per_subject.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

s2c() { $PY code/audit_seed_inference.py --stage cell --semantics pool \
    --max-trials inf \
    --perm "$B_BUDGET" --seeds "$S_ALL" --cells "$OCC_CELLS" \
    --out "$R/budget_audit_all.json" --n_jobs "$NJOBS" --sided "$SIDED"; }

s3a() { $PY code/audit_seed_inference.py --stage calib \
    --reps "$N_REP" --perm "$N_PERM_CAL" --blocks "$N_BLOCKS" --seeds 10 \
    --calib-cell "$CALIB_CELL" \
    --out "$R/seed_inference_audit_calib.json" --n_jobs "$NJOBS"; }

s3b() { $PY code/audit_seed_inference.py --stage calib --unit within \
    --reps "$N_REP" --perm "$N_PERM_CAL" --blocks "$N_BLOCKS" --seeds 10 \
    --calib-cell "$CALIB_CELL" \
    --out "$R/seed_inference_audit_calib_within.json" --n_jobs "$NJOBS"; }

# ---- 第二轮补跑阶段 ------------------------------------------------------
f1() { $PY code/audit_seed_inference.py --stage cell \
    --perm "$B_ROI_FAMILY" --seeds "$S_ROI_FAMILY" --cells roi_family \
    --out "$R/seed_inference_audit_roi_family.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

f2() { $PY code/audit_seed_inference.py --stage cell --semantics per_subject \
    --perm "$B_SCALP_BUDGET" --seeds "$S_SCALP_BUDGET" --cells scalp \
    --out "$R/budget_audit_per_subject_scalp.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

f3() { $PY code/audit_seed_inference.py --stage cell --semantics pool \
    --max-trials inf \
    --perm "$B_SCALP_BUDGET" --seeds "$S_SCALP_BUDGET" --cells scalp \
    --out "$R/budget_audit_all_scalp.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

f4() { $PY code/audit_seed_inference.py --stage cell --unit within \
    --perm "$B_ALL" --seeds "$S_ALL" --cells scalp \
    --out "$R/seed_inference_audit_scalp_within.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

f5() { $PY code/audit_seed_inference.py --stage cell --semantics per_subject \
    --perm "$B_HIB" --seeds "$S_ALL" --cells "$OCC_CELLS" \
    --out "$R/budget_audit_per_subject_hib.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"
      $PY code/audit_seed_inference.py --stage cell --semantics pool \
    --max-trials inf \
    --perm "$B_HIB" --seeds "$S_ALL" --cells "$OCC_CELLS" \
    --out "$R/budget_audit_all_hib.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

f6() { $PY code/audit_seed_inference.py --stage cell --unit within \
    --perm "$B_ROI_FAMILY" --seeds "$S_ROI_FAMILY" --cells roi_family \
    --out "$R/seed_inference_audit_roi_family_within.json" --n_jobs "$NJOBS" \
    --sided "$SIDED"; }

# ---- 第三轮补跑：全脑 18-cell 时间窗家族（§1.4）----------------------------
# full 窗 3 cell 已由 s1b 完成（seed_inference_audit.json）；此处补齐其余
# 5 个时间窗 × 3 target = 15 cell，每窗独立输出（断点续跑按窗隔离）。
# 实测单 cell（B=2000, S=5, 64ch pool, n_jobs=20）≈1.38h → 共约 20.8h。
s1d() {
    for w in N1_100-200ms P2_200-300ms P3_300-450ms LPP_400-700ms LATE_700-1000ms; do
        short="${w%%_*}"   # N1 / P2 / P3 / LPP / LATE
        $PY code/audit_seed_inference.py --stage cell \
            --window "$w" --perm "$B_ALL" --seeds "$S_ALL" --cells scalp \
            --out "$R/seed_inference_audit_scalp_${short}.json" \
            --n_jobs "$NJOBS" --sided "$SIDED"
    done
}

posthoc() { $PY code/audit_posthoc_analysis.py --dir "$R"; }

stage="${1:-all}"
echo "==== Array 大修补充实验 | stage=${stage} | $(date '+%F %T') ===="

case "$stage" in
    smoke)   smoke ;;
    s1a)     s1a ;;
    s1b)     s1b ;;
    s1c)     s1c ;;
    s2a)     s2a ;;
    s2b)     s2b ;;
    s2c)     s2c ;;
    s3a)     s3a ;;
    s3b)     s3b ;;
    s1d)     s1d ;;
    f1)      f1 ;;
    f2)      f2 ;;
    f3)      f3 ;;
    f4)      f4 ;;
    f5)      f5 ;;
    f6)      f6 ;;
    posthoc) posthoc ;;
    all)
        s1a; s1b; s1c; s2a; s2b; s2c; s3a; s3b
        f1; f4; f5; f6; s1d; posthoc
        echo "==== 全部阶段完成（f2/f3 未纳入，需显式指定） $(date '+%F %T') ===="
        ;;
    *) echo "未知 stage: $stage（见脚本头部用法）"; exit 1 ;;
esac
