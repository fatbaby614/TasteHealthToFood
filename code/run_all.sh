#!/usr/bin/env bash
# ============================================================
# run_all.sh — TasteHealthToFood 全流程一键复现脚本
# 在 Linux 工作站上完整重跑全部实验，验证论文结果可复现
# ============================================================
# 用法（在项目根目录下执行）：
#   chmod +x code/run_all.sh
#   ./code/run_all.sh                    # 全量复现（内置论文最终运行参数）
#   ./code/run_all.sh --quick            # 冒烟模式（少量被试，先验证流程能跑通）
#   ./code/run_all.sh --only 1,5,6       # 只跑指定阶段（编号或名称，逗号分隔）
#   ./code/run_all.sh --skip 7           # 跳过指定阶段
#   ./code/run_all.sh --force-preproc    # 忽略 epochs 缓存，从原始数据重新预处理
#   ./code/run_all.sh --svr-perm 100 --svr-seeds 5   # 覆盖置换参数（默认即论文值）
#   ./code/run_all.sh -h                 # 帮助
#
# 阶段划分（--only / --skip 可用编号或名称）：
#   0 envcheck    环境与数据检查（Python 包及版本、数据集路径、epochs 缓存）
#   1 behavior    行为分析：load_behavior / compute_scsr / baseline_model / ddm_model
#   2 preproc     EEG 预处理：eeg_preprocessing（已有缓存自动跳过）/ attach_ratings / eeg_qc_report
#   3 univariate  ERP + 时频：erp_analysis / timefreq_analysis
#   4 integrate   神经-行为整合：sullivan_integration
#   5 decoding    解码：riemannian_decoding / sliding_window_decoding / svr_decoding
#                 + 18-cell 族 v2 多 seed 置换（svr_v2_multiseed_only 并行 6 窗
#                   → svr_v2_merge 合并 + 全族 FDR；M1 决议 2026-09）
#   6 extended    ROI 与交叉分析：svr_roi_channels(论文参数) / svr_x_ddm_ind_diff
#                 / svr_x_scsr_groups / cross_task_analysis
#   7 audit       置换检验稳定性审计：verify_temporal_perm（冒烟模式用 _smoke 版）
#   8 check       产物完整性检查（对照论文 32+2 个结果文件）
#
# 论文复现关键参数（与 outputs/results/svr_roi_channels_summary.json 记录一致）：
#   svr_roi_channels.py       --perm=100 --perm-seeds=5 --max-trials=200
#   svr_decoding.py           --perm=100
#   svr_v2_multiseed_only.py  --perm=100 --seeds=5 --max-trials=200（每窗 1 进程并行）
#   svr_x_scsr_groups.py      --n 106
#   18-cell v2 置换的 FDR 以跨 seed 平均 perm_p 为输入（与 ROI v2 口径一致）
#
# 预计耗时（106 被试全量，工作站）：
#   阶段1 DDM（HSSM 逐被试采样）数小时；阶段2 首次预处理数小时（有缓存则跳过）；
#   阶段5/6 最长：svr_roi_channels（5 seeds × 100 perm）约 2-3 小时；
#   svr_decoding 的 v2 多 seed 置换（6 窗并行，本机实测约 9 小时、
#   工作站预计 3-6 小时）——建议 tmux 过夜运行；svr_x_scsr_groups 约 1-2 小时；
#   阶段7 约 0.5-1 小时。
#   建议 tmux 中运行：tmux new -s rerun './code/run_all.sh'
#   数据集路径：默认自动探测 /mnt/data1/home/tanhuang/datasets/foodeeg-ds007012，
#   或提前 export DATA_DIR=/你的/数据集路径
# ============================================================

set -eo pipefail

# ---------- 参数解析 ----------
QUICK=0
FORCE_PREPROC=0
ONLY=""
SKIP=""
SVR_PERM=100
SVR_SEEDS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)        QUICK=1; shift ;;
        --force-preproc) FORCE_PREPROC=1; shift ;;
        --only)         ONLY="$2"; shift 2 ;;
        --skip)         SKIP="$2"; shift 2 ;;
        --svr-perm)     SVR_PERM="$2"; shift 2 ;;
        --svr-seeds)    SVR_SEEDS="$2"; shift 2 ;;
        -h|--help)      grep -E "^# " "$0" | sed 's/^# //'; exit 0 ;;
        *) echo "[ERROR] 未知参数: $1（-h 查看帮助）"; exit 1 ;;
    esac
done

# ---------- 路径与环境 ----------
# 脚本位于 code/ 目录，项目根是其上一级目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="outputs/logs/run_all_${TIMESTAMP}"
mkdir -p "$LOGDIR"
MASTER_LOG="$LOGDIR/master.log"
exec &> >(tee "$MASTER_LOG")

trap 'echo ""; echo "[ERROR] 脚本在第 $LINENO 行附近中断，日志：$LOGDIR"' ERR

# 数据集路径：config.py 读 DATA_DIR，eeg_preprocessing.py 读 FOODEEG_DATASET_DIR，
# 两者统一设置，避免预处理找不到数据
DEFAULT_DS="/mnt/data1/home/tanhuang/datasets/foodeeg-ds007012"
if [[ -z "${DATA_DIR:-}" && -d "$DEFAULT_DS" ]]; then
    export DATA_DIR="$DEFAULT_DS"
fi
if [[ -n "${DATA_DIR:-}" ]]; then
    export FOODEEG_DATASET_DIR="${FOODEEG_DATASET_DIR:-$DATA_DIR}"
fi
export MPLBACKEND=Agg

PY="python"
command -v python >/dev/null 2>&1 || PY="python3"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[ERROR] 未找到 python，请先激活conda 环境"; exit 1
fi

# ---------- 通用函数 ----------
stage_enabled() {
    local id="$1" name="$2"
    if [[ -n "$ONLY" ]]; then
        [[ ",$ONLY," == *",$id,"* || ",$ONLY," == *",$name,"* ]] && return 0 || return 1
    fi
    if [[ -n "$SKIP" ]]; then
        [[ ",$SKIP," == *",$id,"* || ",$SKIP," == *",$name,"* ]] && return 1
    fi
    return 0
}

banner() {
    echo ""
    echo "############################################################"
    echo "# $1"
    echo "############################################################"
}

run_step() {
    local step_id="$1"; shift
    local script="$1"; shift
    local log="${LOGDIR}/${step_id}_${script%.py}.log"
    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN] $step_id: $PY -u code/$script $*"
    echo "[LOG] $log"
    local t0=$(date +%s)
    if $PY -u "code/$script" "$@" 2>&1 | tee "$log"; then
        local t1=$(date +%s)
        echo "[OK] $step_id ($script) 完成，用时 $((t1-t0))s"
        echo "OK $((t1-t0))s ${step_id} ${script} $*" >> "$LOGDIR/summary.txt"
    else
        local rc=$?
        echo "[FAIL] $step_id ($script) 退出码 $rc，查看日志：$log"
        echo "FAIL rc=$rc ${step_id} ${script} $*" >> "$LOGDIR/summary.txt"
        exit "$rc"
    fi
}

# ---------- 开场信息 ----------
banner "TasteHealthToFood 全流程复现  $(date '+%F %T')"
if [[ $QUICK -eq 1 ]]; then MODE="冒烟 (--quick)"; else MODE="全量 (论文参数)"; fi
echo "模式        : $MODE"
echo "Python      : $($PY --version 2>&1)"
echo "项目根目录  : $PROJECT_ROOT"
echo "DATA_DIR    : ${DATA_DIR:-未设置（config.py 自动探测）}"
echo "FOODEEG_DATASET_DIR : ${FOODEEG_DATASET_DIR:-未设置}"
echo "置换参数    : svr_perm=$SVR_PERM, svr_seeds=$SVR_SEEDS"
echo "日志目录    : $LOGDIR"
echo "" > "$LOGDIR/summary.txt"

# ============================================================
# 阶段 0：环境与数据检查
# ============================================================
if stage_enabled 0 envcheck; then
    banner "阶段 0：环境与数据检查"
    $PY - <<'PYEOF' 2>&1 | tee "$LOGDIR/00_envcheck.log"
import importlib, sys, platform
print('Python :', sys.version.split()[0], '|', platform.platform())
REQUIRED = ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels', 'mne',
            'sklearn', 'pyriemann', 'hssm', 'pymc', 'arviz']
missing = []
for name in REQUIRED:
    try:
        m = importlib.import_module(name)
        print(f'  {name:12s} {getattr(m, "__version__", "?")}')
    except Exception as e:
        missing.append(name)
        print(f'  {name:12s} 缺失 ({e.__class__.__name__})')
if missing:
    print('缺失包:', ', '.join(missing))
    sys.exit(1)
print('环境检查通过')
PYEOF
    if [[ -n "${DATA_DIR:-}" && ! -d "$DATA_DIR" ]]; then
        echo "[WARN] DATA_DIR 不存在: $DATA_DIR"
    fi
    EPOCH_N=$(ls data/epochs/*_stim_epochs.npy 2>/dev/null | wc -l || true)
    LABEL_N=$(ls data/epochs/*_stim_labels_with_ratings.npz 2>/dev/null | wc -l || true)
    echo "epochs 缓存          : ${EPOCH_N} 个被试"
    echo "labels_with_ratings  : ${LABEL_N} 个被试"
    if [[ "$EPOCH_N" -eq 0 ]]; then
        echo "[提示] 无 epochs 缓存，阶段 2 将从原始数据完整预处理（耗时数小时）"
    fi
    echo "[OK] 阶段 0 完成"
fi

# ============================================================
# 阶段 1：行为分析
# ============================================================
if stage_enabled 1 behavior; then
    banner "阶段 1：行为分析"
    run_step 1a load_behavior.py
    run_step 1b compute_scsr.py
    run_step 1c baseline_model.py
    if [[ $QUICK -eq 1 ]]; then
        run_step 1d ddm_model.py --quick
    else
        run_step 1d ddm_model.py
    fi
    echo "[OK] 阶段 1 完成"
fi

# ============================================================
# 阶段 2：EEG 预处理
# ============================================================
if stage_enabled 2 preproc; then
    banner "阶段 2：EEG 预处理"
    PREPROC_ARGS=()
    [[ $QUICK -eq 1 ]] && PREPROC_ARGS+=(--quick)
    [[ $FORCE_PREPROC -eq 1 ]] && PREPROC_ARGS+=(--force)
    run_step 2a eeg_preprocessing.py "${PREPROC_ARGS[@]}"
    run_step 2b attach_ratings.py
    run_step 2c eeg_qc_report.py
    echo "[OK] 阶段 2 完成"
fi

# ============================================================
# 阶段 3：ERP + 时频
# ============================================================
if stage_enabled 3 univariate; then
    banner "阶段 3：ERP 与时频分析"
    run_step 3a erp_analysis.py
    run_step 3b timefreq_analysis.py
    echo "[OK] 阶段 3 完成"
fi

# ============================================================
# 阶段 4：神经-行为整合
# ============================================================
if stage_enabled 4 integrate; then
    banner "阶段 4：神经-行为整合"
    run_step 4a sullivan_integration.py
    echo "[OK] 阶段 4 完成"
fi

# ============================================================
# 阶段 5：解码分析
# ============================================================
if stage_enabled 5 decoding; then
    banner "阶段 5：解码分析（Riemannian / 滑窗 / SVR + v2 多 seed 置换）"
    if [[ $QUICK -eq 1 ]]; then
        run_step 5a riemannian_decoding.py --quick
        run_step 5b sliding_window_decoding.py --n=10
        run_step 5c svr_decoding.py --quick --perm=10
        # 冒烟：仅 LPP 单窗 2 seeds × 3 perm，验证 v2 多 seed 链路可跑通
        run_step 5d svr_v2_multiseed_only.py --perm=3 --seeds=2 --max-trials=60 --windows=LPP_400-700ms
    else
        run_step 5a riemannian_decoding.py
        run_step 5b sliding_window_decoding.py
        run_step 5c svr_decoding.py "--perm=$SVR_PERM"
        # 5d：18-cell 族 v2 多 seed 置换（M1 决议 2026-09；与 ROI v2 同口径：
        # 池级 budget 200、合并池打乱、FDR 输入=跨 seed 平均 perm_p）。
        # 6 个时间窗各起 1 个进程并行，全部完成后再合并。
        echo ""
        echo "------------------------------------------------------------"
        echo "[RUN] 5d: v2 多 seed 置换（每窗 $SVR_SEEDS seeds × $SVR_PERM perm，池级 budget 200）"
        MS_LOG_DIR="$LOGDIR/5d_v2_multiseed"
        mkdir -p "$MS_LOG_DIR"
        MS_PIDS=()
        for wname in N1_100-200ms P2_200-300ms P3_300-450ms LPP_400-700ms LATE_700-1000ms full_0-1000ms; do
            tag="${wname%%_*}"
            $PY -u code/svr_v2_multiseed_only.py --perm="$SVR_PERM" --seeds="$SVR_SEEDS" \
                --max-trials=200 --windows="$wname" > "$MS_LOG_DIR/ms_${tag}.log" 2>&1 &
            MS_PIDS+=("$!")
            echo "  [launch] $tag (pid $!)"
        done
        MS_FAIL=0
        for pid in "${MS_PIDS[@]}"; do
            wait "$pid" || MS_FAIL=1
        done
        if [[ $MS_FAIL -eq 1 ]]; then
            echo "[FAIL] 5d 部分窗口置换失败，查看日志：$MS_LOG_DIR"
            exit 1
        fi
        echo "[OK] 5d 完成（6 窗 v2 置换均落盘）"
        # 5e：合并 6 窗 → 18 格最终文件 + 全族 FDR
        run_step 5e svr_v2_merge.py
    fi
    echo "[OK] 阶段 5 完成"
fi

# ============================================================
# 阶段 6：ROI 与交叉分析（论文核心结果）
# ============================================================
if stage_enabled 6 extended; then
    banner "阶段 6：ROI 与交叉分析（论文核心结果）"
    if [[ $QUICK -eq 1 ]]; then
        run_step 6a svr_roi_channels.py --quick --perm=10 --perm-seeds=1 --max-trials=20
        run_step 6b svr_x_ddm_ind_diff.py --quick
        run_step 6c svr_x_scsr_groups.py --quick
        run_step 6d cross_task_analysis.py
    else
        run_step 6a svr_roi_channels.py "--perm=$SVR_PERM" "--perm-seeds=$SVR_SEEDS" --max-trials=200
        run_step 6b svr_x_ddm_ind_diff.py
        run_step 6c svr_x_scsr_groups.py --n 106
        run_step 6d cross_task_analysis.py
    fi
    echo "[OK] 阶段 6 完成"
fi

# ============================================================
# 阶段 7：置换检验稳定性审计
# ============================================================
if stage_enabled 7 audit; then
    banner "阶段 7：置换检验稳定性审计"
    if [[ $QUICK -eq 1 ]]; then
        run_step 7a verify_temporal_perm_smoke.py
    else
        run_step 7a verify_temporal_perm.py
    fi
    echo "[OK] 阶段 7 完成"
fi

# ============================================================
# 阶段 8：产物完整性检查
# ============================================================
if stage_enabled 8 check && [[ $QUICK -eq 0 ]]; then
    banner "阶段 8：产物完整性检查"
    EXPECTED=(
        baseline_summary.json
        cross_task_correlations.csv
        cross_task_summary.json
        cutpoint1_summary.json
        ddm_regression_params.csv
        ddm_simple_params.csv
        ddm_summary.json
        erp_roi_ttests.csv
        erp_summary.json
        erp_window_means.csv
        individual_metrics.csv
        neural_behavior_correlations.csv
        riemannian_decoding_results.csv
        riemannian_summary.json
        scsr_per_subject.csv
        sliding_window_decoding_results.csv
        sliding_window_stats.csv
        sliding_window_summary.json
        svr_ddm_correlations.csv
        svr_ddm_correlations.json
        svr_decoding_results.csv
        svr_decoding_multiseed.csv
        svr_decoding_multiseed.json
        svr_roi_channels_results.csv
        svr_roi_channels_summary.csv
        svr_roi_channels_summary.json
        svr_scsr_groups_results.csv
        svr_scsr_groups_summary.csv
        svr_scsr_groups_summary.json
        svr_summary.json
        tfr_band_ttests.csv
        tfr_band_window_means.csv
        tfr_group_db.npy
        tfr_group_raw.npy
    )
    MISSING=0
    for f in "${EXPECTED[@]}"; do
        if [[ -f "outputs/results/$f" ]]; then
            echo "  [OK]      $f"
        else
            echo "  [MISSING] $f"
            MISSING=$((MISSING+1))
        fi
    done
    echo ""
    echo "产物检查：共 ${#EXPECTED[@]} 个文件，缺失 $MISSING 个"
    echo "产物检查：共 ${#EXPECTED[@]} 个文件，缺失 $MISSING 个" >> "$LOGDIR/summary.txt"
    if [[ $MISSING -gt 0 ]]; then
        echo "[FAIL] 存在缺失产物，请检查对应阶段日志"
        exit 1
    fi
    echo "[OK] 阶段 8 完成"
elif stage_enabled 8 check && [[ $QUICK -eq 1 ]]; then
    echo "[SKIP] 阶段 8（冒烟模式不检查全量产物）"
fi

# ---------- 汇总 ----------
sleep 1
banner "全部完成 $(date '+%F %T')"
echo "模式   : $MODE"
echo "日志   : $LOGDIR"
echo ""
echo "各步骤用时与状态（summary.txt）："
cat "$LOGDIR/summary.txt"
echo ""
echo "结果目录：outputs/results/（32 个论文结果文件）、outputs/figures/"
echo "复现核对建议：对比 svr_roi_channels_summary.json / svr_summary.json"
echo "/ riemannian_summary.json / ddm_summary.json / svr_scsr_groups_summary.json"
echo "与论文报告数值是否一致"
