#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HFT_LSTM_train_POST30_ms_torch_multitask.py
v2025-09-18 (robust)

Summary
- Trains a multi-task model that predicts, per horizon:
  (1) Regression (bps): y_reg_bps = 1e4 * (log(mid_{t+h}) - log(mid_t))
  (2) Direction classification {-1,0,+1} from the TRUE decimal delta with a dead-zone eps:
        eps_dec = cls_eps_bps / 1e4
        dir = +1 if y_dec >  eps_dec
            =  0 if |y_dec| <= eps_dec
            = -1 if y_dec < -eps_dec
- Anchors every --step_ms (default 30ms). Optional PT-window filter [--release_time_pt, +--post_window_min).
- After training, runs an in-month evaluation and appends a summary row to CSV.

Key CSV Metrics (overall + per-horizon)
- MAE/MSE/R2: regression in bps / bps²
- HIT_REG: 1{sign(pred_reg_dec) == sign(true_dec)}  (fractions in [0,1])
- If --has_cls:
    HITCLS_3C: 3-class accuracy vs {-1,0,+1}
    HITCLS_2C: accuracy on non-flat cases only (|true|>eps); excludes zero class
    COV_2C:    fraction of non-flat cases (coverage)
- CSV also includes epochs, batch_size, note, model_tag (saved file name).

Loss
- total = lam_reg * SmoothL1(pred_bps, true_bps) + lam_cls * CE(logits, dir_labels)
- Optional class weights via --cls_weights w_down w_flat w_up
- --cls_ignore_zero (CE ignores the 0 class) is supported.

OOM Robustness
- Mixed precision (AMP) is enabled on CUDA to reduce memory.
- Use --win_b, --batch_anchors, --batch_size, --max_seq_len to control memory.
- Export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.

I/O
- Saves:  save_dir/model_<YYYYMM>_<arch>.pt
           save_dir/model_<YYYYMM>_<arch>.json
  JSON includes has_cls, cls_eps_bps, normalization, horizons, step_ms, windows, etc.

========================================
Event-driven monthly trainin
========================================
SYMBOL="ZN"; PARENT="${SYMBOL}.FUT"
RUN_ROOT="/home/ubuntu/HFT_forecast/fit_data/roll12mo/jobs"
SAVE_DIR="$RUN_ROOT/202410_202509/${SYMBOL}"
SUMMARY_CSV="$RUN_ROOT/summary_roll12_202410_202509_${SYMBOL}.csv"
PY=$(which python)

CUDA_VISIBLE_DEVICES=1 nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_a.py \
  --data_roots /home/ubuntu/HFT_forecast/model_data/2024 /home/ubuntu/HFT_forecast/model_data/2025 \
  --event_type jobs \
  --instrument "${SYMBOL}" \
  --parent_symbol "${PARENT}" \
  --save_dir   "${SAVE_DIR}" \
  --summary_csv "${SUMMARY_CSV}" \
  --dates 202410 202411 202412 202501 202502 202503 202504 202505 202506 202507 202508 202509 \
  --note "JOBS|${SYMBOL}|ROLL12(2024-10..2025-09); PT 05:30+10m; win_a=0 win_b=15000; max_seq=15000; horizons=30/60/120/250/500/750/1000ms; multitask(reg+cls); target=logret y_scale=1.0" \
  --epochs 6 --batch_size 1024 --win_a 0 --win_b 15000 --batch_anchors 512 --sample_size 0 \
  --step_ms 30 --horizons_ms 30 60 120 250 500 750 1000 \
  --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.0 \
  --restrict_to_release_window --tod_start_pt 05:30 --post_window_min 30 \
  --max_seq_len 15000 \
  --target logret --y_scale 1.0 \
  --has_cls --cls_eps_bps 0.5 --lam_reg 1.0 --lam_cls 1.0 --cls_ignore_zero \
  > "${SAVE_DIR}/train_roll12_jobs_${SYMBOL}_202410_202509_v1_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

========================================
Daily HFT trainin Rolling last 30 trading days
========================================

>>>>>> Rolling last 30 trading days for ES

PY=${PY:-python3}
LOGDIR="/home/ubuntu/HFT_forecast/logs_hft/train"
SAVE_ROOT="/home/ubuntu/HFT_forecast/fit_data/roll30/live"
DATA_2025="/home/ubuntu/HFT_forecast/model_data/2025/daily"
mkdir -p "$LOGDIR" "$SAVE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

INSTRUMENT="ES"; PROFILE="hft"
nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_b.py \
  --data_roots "$DATA_2025" \
  --event_type live \
  --profile "$PROFILE" --instrument "$INSTRUMENT" --parent_symbol "$INSTRUMENT.FUT" \
  --save_dir "$SAVE_ROOT" \
  --summary_csv "$SAVE_ROOT/summary_roll30_live_${INSTRUMENT}_${PROFILE}.csv" \
  --rolling_days 30 --exclude_today \
  --accept_daily_single \
  --epochs 4 --batch_size 256 \
  --win_a 0 --win_b 60000 \
  --batch_anchors 256 \
  --sample_size 600000 \
  --step_ms 1500 \
  --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.35 \
  --target logret --y_scale 10000 \
  --lam_reg 1.0 \
  > "$LOGDIR/train_roll30_live_${INSTRUMENT}_${PROFILE}_reg_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

>>>>>> Rolling last 30 trading days for ZN

PY=${PY:-python3}
LOGDIR="/home/ubuntu/HFT_forecast/logs_hft/train"
SAVE_ROOT="/home/ubuntu/HFT_forecast/fit_data/roll30/live"
DATA_2025="/home/ubuntu/HFT_forecast/model_data/2025/daily"
mkdir -p "$LOGDIR" "$SAVE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

INSTRUMENT="ZN"; PROFILE="hft"
nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_b.py \
  --data_roots "$DATA_2025" \
  --event_type live \
  --profile "$PROFILE" --instrument "$INSTRUMENT" --parent_symbol "$INSTRUMENT.FUT" \
  --save_dir "$SAVE_ROOT" \
  --summary_csv "$SAVE_ROOT/summary_roll30_live_${INSTRUMENT}_${PROFILE}.csv" \
  --rolling_days 30 --exclude_today \
  --accept_daily_single \
  --epochs 4 --batch_size 288 \
  --win_a 0 --win_b 120000 \
  --batch_anchors 288 \
  --sample_size 600000 \
  --step_ms 3000 \
  --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.2 \
  --target logret --y_scale 1.0 \
  --lam_reg 1.0 \
  > "$LOGDIR/train_roll30_live_${INSTRUMENT}_${PROFILE}_reg_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &


>>>>>> Exact date range

# ES (GPU 0): 2025-09-22 .. 2025-10-13
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_a.py \
  --data_roots "$DATA_2025" \
  --event_type live --instrument ES --parent_symbol ES.FUT \
  --save_dir   "$SAVE_ROOT/ES_range" \
  --summary_csv "$SAVE_ROOT/ES_range/summary_20250922_20251013_ES.csv" \
  --start_date 2025-09-22 --end_date 2025-10-13 --accept_daily_single \
  --restrict_to_release_window --release_time_pt 05:30 --post_window_min 30 \
  --epochs 6 --batch_size 1024 --win_a 0 --win_b 15000 --batch_anchors 512 \
  --step_ms 30 --horizons_ms 30 60 120 250 500 750 1000 \
  --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.0 \
  --target logret --y_scale 1.0 \
  --has_cls --cls_eps_bps 0.5 --lam_reg 1.0 --lam_cls 1.0 --cls_ignore_zero \
  > "$LOGDIR/train_range_ES_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

# ZN (GPU 1) same range
CUDA_VISIBLE_DEVICES=1 nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_a.py \
  --data_roots "$DATA_2025" \
  --event_type live --instrument ZN --parent_symbol ZN.FUT \
  --save_dir   "$SAVE_ROOT/ZN_range" \
  --summary_csv "$SAVE_ROOT/ZN_range/summary_20250922_20251013_ZN.csv" \
  --start_date 2025-09-22 --end_date 2025-10-13 --accept_daily_single \
  --restrict_to_release_window --release_time_pt 05:30 --post_window_min 30 \
  --epochs 6 --batch_size 1024 --win_a 0 --win_b 15000 --batch_anchors 512 \
  --step_ms 30 --horizons_ms 30 60 120 250 500 750 1000 \
  --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.0 \
  --target logret --y_scale 1.0 \
  --has_cls --cls_eps_bps 0.5 --lam_reg 1.0 --lam_cls 1.0 --cls_ignore_zero \
  > "$LOGDIR/train_range_ZN_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

========================================
Daily IDT (intra-day trading) trainin
========================================

>>>>> lean coverage to cover idt horizons

PY=${PY:-python3}
LOGDIR="/home/ubuntu/HFT_forecast/logs_hft/train"
SAVE_ROOT="/home/ubuntu/HFT_forecast/fit_data/roll30/live"
DATA_2025="/home/ubuntu/HFT_forecast/model_data/2025/daily"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

INSTRUMENT="ES"; PROFILE="idt"

nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_b.py \
  --data_roots "$DATA_2025" \
  --event_type live \
  --profile "$PROFILE" \
  --instrument "$INSTRUMENT" --parent_symbol "$INSTRUMENT.FUT" \
  --book_features auto \
  --book_max_level 10 \
  --save_dir "$SAVE_ROOT" \
  --summary_csv "$SAVE_ROOT/summary_roll30_live_${INSTRUMENT}_${PROFILE}_L23.csv" \
  --rolling_days 30 --exclude_today \
  --accept_daily_single \
  --epochs 4 --batch_size 288 \
  --win_a 0 --win_b 90000 \
  --batch_anchors 288 \
  --sample_size 600000 \
  --step_ms 3000 \
  --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.2 \
  --target logret --y_scale 1.0 \
  --lam_reg 1.0 \
  > "$LOGDIR/train_roll30_live_${INSTRUMENT}_${PROFILE}_L23_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

PY=${PY:-python3}
LOGDIR="/home/ubuntu/HFT_forecast/logs_hft/train"
SAVE_ROOT="/home/ubuntu/HFT_forecast/fit_data/roll30/live"
DATA_2025="/home/ubuntu/HFT_forecast/model_data/2025/daily"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

INSTRUMENT="ZN"; PROFILE="idt"

nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_b.py \
  --data_roots "$DATA_2025" \
  --event_type live \
  --profile "$PROFILE" \
  --instrument "$INSTRUMENT" --parent_symbol "$INSTRUMENT.FUT" \
  --book_features auto \
  --book_max_level 10 \
  --save_dir "$SAVE_ROOT" \
  --summary_csv "$SAVE_ROOT/summary_roll30_live_${INSTRUMENT}_${PROFILE}_L23.csv" \
  --rolling_days 30 --exclude_today \
  --accept_daily_single \
  --epochs 4 --batch_size 288 \
  --win_a 0 --win_b 90000 \
  --batch_anchors 288 \
  --sample_size 600000 \
  --step_ms 3000 \
  --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.2 \
  --target logret --y_scale 1.0 \
  --lam_reg 1.0 \
  > "$LOGDIR/train_roll30_live_${INSTRUMENT}_${PROFILE}_L23_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

watch -n1 nvidia-smi
pgrep -a python | grep HFT_LSTM_train_POST30
tail -f "$CPI_CALM_DIR"/train_cpi_calm_*.log
tail -f "$CPI_VOLATILE_DIR"/train_cpi_volatile_*.log

>>>>> combined commands using loop >>>>>>>>

!!!!! in-sample training ; for L12 training add --prefer_l12 \ !!!!!!!

PY=${PY:-python3}
LOGDIR="/home/ubuntu/HFT_forecast/logs_hft/train"
SAVE_ROOT="/home/ubuntu/HFT_forecast/fit_data/roll30/live"
DATA_2025="/home/ubuntu/HFT_forecast/model_data/2025/daily"
DATA_2026="/home/ubuntu/HFT_forecast/model_data/2026/daily"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROFILE="idt"

INSTRUMENTS=(ES ZN)
GPUS=(0 1)

for idx in "${!INSTRUMENTS[@]}"; do
  INSTRUMENT="${INSTRUMENTS[$idx]}"
  GPU="${GPUS[$((idx % ${#GPUS[@]}))]}"

  export CUDA_VISIBLE_DEVICES="$GPU"

  echo "[INFO] Launching ${INSTRUMENT}/${PROFILE} on GPU ${GPU}"

  nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_b.py \
    --data_roots "$DATA_2025" "$DATA_2026" \
    --event_type live \
    --profile "$PROFILE" \
    --instrument "$INSTRUMENT" --parent_symbol "$INSTRUMENT.FUT" \
    --book_features auto \
    --book_max_level 1 \
    --save_dir "$SAVE_ROOT" \
    --summary_csv "$SAVE_ROOT/summary_roll30_live_${INSTRUMENT}_${PROFILE}_L1.csv" \
    --rolling_days 30 --exclude_today \
    --accept_daily_single \
    --epochs 4 --batch_size 288 \
    --win_a 0 --win_b 90000 \
    --batch_anchors 288 \
    --sample_size 600000 \
    --step_ms 3000 \
    --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.2 \
    --target logret --y_scale 1.0 \
    --lam_reg 1.0 \
    > "$LOGDIR/train_roll30_live_${INSTRUMENT}_${PROFILE}_L1_gpu${GPU}_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &
done

!!!!! in-sample training ; ZN uses rolling 60 ES using rolling 30 !!!!!!!

PY=${PY:-python3}
LOGDIR="/home/ubuntu/HFT_forecast/logs_hft/train"
SAVE_ROOT="/home/ubuntu/HFT_forecast/fit_data/roll30/live"
DATA_2025="/home/ubuntu/HFT_forecast/model_data/2025/daily"
DATA_2026="/home/ubuntu/HFT_forecast/model_data/2026/daily"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROFILE="hft"

INSTRUMENTS=(ES ZN)
GPUS=(0 1)

for idx in "${!INSTRUMENTS[@]}"; do
  INSTRUMENT="${INSTRUMENTS[$idx]}"
  GPU="${GPUS[$((idx % ${#GPUS[@]}))]}"

  # defaults (ES)
  ROLLING_DAYS=30
  EPOCHS=4
  STEP_MS=3000
  SAMPLE_SIZE=600000
  BATCH_ANCHORS=288
  LOGTAG="roll30"

  # overrides (ZN)
  if [[ "$INSTRUMENT" == "ZN" ]]; then
    ROLLING_DAYS=60
    EPOCHS=5
    STEP_MS=5000
    SAMPLE_SIZE=900000
    BATCH_ANCHORS=384
    LOGTAG="roll60"
  fi

  export CUDA_VISIBLE_DEVICES="$GPU"

  echo "[INFO] Launching ${INSTRUMENT}/${PROFILE} on GPU ${GPU} (rolling_days=${ROLLING_DAYS}, epochs=${EPOCHS}, step_ms=${STEP_MS}, sample_size=${SAMPLE_SIZE}, batch_anchors=${BATCH_ANCHORS})"

  nohup "$PY" -u /home/ubuntu/HFT_forecast/HFT_LSTM_train_POST30_ms_torch_multitask_b.py \
    --data_roots "$DATA_2025" "$DATA_2026" \
    --event_type live \
    --profile "$PROFILE" \
    --instrument "$INSTRUMENT" --parent_symbol "$INSTRUMENT.FUT" \
    --book_features auto \
    --book_max_level 1 \
    --save_dir "$SAVE_ROOT" \
    --summary_csv "$SAVE_ROOT/summary_${LOGTAG}_live_${INSTRUMENT}_${PROFILE}_L1.csv" \
    --rolling_days "$ROLLING_DAYS" --exclude_today \
    --accept_daily_single \
    --epochs "$EPOCHS" --batch_size 288 \
    --win_a 0 --win_b 90000 \
    --batch_anchors "$BATCH_ANCHORS" \
    --sample_size "$SAMPLE_SIZE" \
    --step_ms "$STEP_MS" \
    --arch lstm --lstm_hidden 64 --lstm_layers 2 --lstm_dropout 0.2 \
    --target logret --y_scale 1.0 \
    --lam_reg 1.0 \
    > "$LOGDIR/train_${LOGTAG}_live_${INSTRUMENT}_${PROFILE}_L1_gpu${GPU}_$(date -u +%Y%m%d_%H%M%S).log" 2>&1 &

done

"""

from __future__ import annotations
import os, re, gc, csv, glob, sys, time, math, argparse, json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
import datetime
import hashlib

print("[BANNER] HFT_LSTM_train_POST30_ms_torch_multitask (ms-horizons + PT-window + AMP) v2025-10-14-rollwin", flush=True)

CLIP_NORM = 1.0
t0 = time.time()

# ========================= Profiles & defaults =========================
HFT_DEFAULTS = [30, 60, 120, 250, 500, 750, 1000] # up to 1sec
IDT_DEFAULTS = [
    2000, 5000, 10000, 30000, 60000, 120000,
    300000, 480000, 600000, 900000, 1200000, 1800000, 3600000 ] # up to 60min

# ========================= Helpers: summary header =========================
def build_train_summary_header(horizons_ms: List[int], has_cls: bool) -> List[str]:
    base = ["date","mode","n","MAE","MSE","R2","HIT_REG","epochs","batch_size","train_stamp","note","model_tag"]
    per_h = []
    for h in horizons_ms:
        per_h += [f"MAE_{h}ms", f"MSE_{h}ms", f"HIT_REG_{h}ms"]
    cols = base[:6] + per_h + base[6:]
    if has_cls:
        cls_overall = ["HITCLS_3C","HITCLS_2C","COV_2C"]
        cls_per_h = []
        for h in horizons_ms:
            cls_per_h += [f"HITCLS_3C_{h}ms", f"HITCLS_2C_{h}ms", f"COV_2C_{h}ms"]
        cols = cols[:7] + cls_overall + cols[7:]
        cols = cols[:6 + 3*len(horizons_ms)] + cls_per_h + cols[6 + 3*len(horizons_ms):]
    return cols

def append_summary(summary_csv: str, header: List[str], row: dict):
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    header_exists = os.path.exists(summary_csv) and os.path.getsize(summary_csv) > 0
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not header_exists:
            w.writeheader()
        out = {k: row.get(k, "") for k in header}
        w.writerow(out)

# ========================= I/O helpers =========================
def device_of_choice():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={dev}, cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        try:
            print(f"[INFO] cuda_device_count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}", flush=True)
        except Exception:
            pass
    return dev

def yyyymm_from_path(p: str) -> str:
    m = re.search(r'(20\d{2})(0[1-9]|1[0-2])', os.path.basename(p))
    if not m:
        m = re.search(r'(20\d{2})(0[1-9]|1[0-2])', p)
    return m.group(0) if m else "unknown"

def yyyymmdd_from_path(p: str) -> Optional[str]:
    m = re.search(r'(\d{4})(\d{2})(\d{2})', os.path.basename(p))
    if not m:
        m = re.search(r'(\d{4})(\d{2})(\d{2})', p)
    if not m:
        return None
    yyyy, mm, dd = m.groups()
    try:
        pd.Timestamp(f"{yyyy}-{mm}-{dd}")
        return f"{yyyy}{mm}{dd}"
    except Exception:
        return None

def load_month_files(
    root_or_roots: List[str] | str,
    dates: Optional[List[str]],
    event_type: Optional[str] = None,
    prefer_l12: bool = False,   # True: try *_L12_SLIM first, else fallback to *_L1_SLIM
) -> List[str]:
    roots = root_or_roots if isinstance(root_or_roots, list) else [root_or_roots]
    paths: List[str] = []

    def _filter(got: List[str]) -> List[str]:
        if dates:
            dset = set(dates)
            got = [p for p in got if any(d in p for d in dset)]
        if event_type:
            got = [p for p in got if os.path.basename(p).startswith(f"{event_type}_")]
        return got

    for r in roots:
        got: List[str] = []

        # 1) Prefer L12 if requested
        if prefer_l12:
            got = glob.glob(os.path.join(r, "**", "*_L12_SLIM.parquet"), recursive=True)
            got = _filter(got)

        # 2) Fall back to L1
        if not got:
            got = glob.glob(os.path.join(r, "**", "*_L1_SLIM.parquet"), recursive=True)
            got = _filter(got)

        # 3) Final fallback: any *_SLIM.parquet (keeps old behavior resilient)
        if not got:
            got = glob.glob(os.path.join(r, "**", "*_SLIM.parquet"), recursive=True)
            got = _filter(got)

        paths.extend(got)

    return sorted(set(paths))

def read_ticks_df(path: str, ts_col: str = "ts_event", ts_unit: str = "ns") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    idx = None
    if ts_col in df.columns:
        s = df[ts_col]
        if pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce", utc=True); idx = s.dt.tz_convert(None)
        elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            idx = pd.to_datetime(s, unit=ts_unit, errors="coerce", utc=True).tz_convert(None)
        else:
            s = pd.to_datetime(s, errors="coerce", utc=True); idx = s.dt.tz_convert(None)

    if idx is None:
        for c in ("timestamp","datetime","time","ts"):
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                idx = s.dt.tz_convert(None)
                break

    if idx is None or idx.isna().all():
        raise ValueError(f"Could not find usable timestamp in {path} (looked for {ts_col})")

    df.index = pd.DatetimeIndex(idx); df.index.name = None
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df

# =========================  Helper: depth (L2) + order-flow (L3) ========================= 

BOOK_LVL_RE = re.compile(r'^(bid|ask)_(px|price|sz|size)_([0-9]+)$', re.IGNORECASE)

def _pick_first(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def augment_with_book_features(
    df: pd.DataFrame,
    max_level: int = 10,
    inst_tag: str | None = None,
) -> pd.DataFrame:
    """
    Derive L2 (depth) and L3-like order-flow features from SLIM parquet.

    L2 expectation (auto-detected, case-insensitive):
      - bid_px_1, bid_px_2, ..., bid_px_N
      - ask_px_1, ask_px_2, ..., ask_px_N
      - bid_sz_1 / bid_size_1, ...
      - ask_sz_1 / ask_size_1, ...

    L3 expectation (any that exist will be used, all optional):
      - order adds / cancels / trades (1s bar aggregates)
      - aggressive buy / sell counts or volumes
      - sweep depth if available

    If none of the relevant columns exist, this is a no-op.
    """
    cols = list(df.columns)

    # ------------- L2: depth aggregation -------------
    levels: Dict[int, Dict[str, str]] = {}

    for c in cols:
        m = BOOK_LVL_RE.match(c)
        if not m:
            continue
        side, kind, lvl_s = m.groups()
        lvl = int(lvl_s)
        if lvl > max_level:
            continue
        key = f"{side.lower()}_{kind.lower()}"
        levels.setdefault(lvl, {})[key] = c

    bid_depth_cols = []
    ask_depth_cols = []
    imb_cols = []
    eps = 1e-9

    if levels:
        lvl_keys = sorted(levels.keys())
        cum_bid = None
        cum_ask = None

        for L in lvl_keys:
            info = levels[L]
            bid_sz_col = info.get("bid_sz") or info.get("bid_size")
            ask_sz_col = info.get("ask_sz") or info.get("ask_size")
            if bid_sz_col is None or ask_sz_col is None:
                continue

            bsz = df[bid_sz_col].astype("float64")
            asz = df[ask_sz_col].astype("float64")

            if cum_bid is None:
                cum_bid = bsz
                cum_ask = asz
            else:
                cum_bid = cum_bid + bsz
                cum_ask = cum_ask + asz

            depth_bid_name = f"depth_bid_L{L}"
            depth_ask_name = f"depth_ask_L{L}"
            imb_name       = f"depth_imb_L{L}"

            df[depth_bid_name] = cum_bid
            df[depth_ask_name] = cum_ask
            df[imb_name]       = (cum_bid - cum_ask) / (cum_bid + cum_ask + eps)

            bid_depth_cols.append(depth_bid_name)
            ask_depth_cols.append(depth_ask_name)
            imb_cols.append(imb_name)

        # Microprice at level 1 if we have prices & sizes
        lvl1 = levels.get(1, {})
        bid_px1 = lvl1.get("bid_px") or lvl1.get("bid_price")
        ask_px1 = lvl1.get("ask_px") or lvl1.get("ask_price")
        bid_sz1 = lvl1.get("bid_sz") or lvl1.get("bid_size")
        ask_sz1 = lvl1.get("ask_sz") or lvl1.get("ask_size")

        if bid_px1 and ask_px1 and bid_sz1 and ask_sz1:
            bp = df[bid_px1].astype("float64")
            ap = df[ask_px1].astype("float64")
            bs = df[bid_sz1].astype("float64")
            as_ = df[ask_sz1].astype("float64")
            micro = (ap * bs + bp * as_) / (bs + as_ + eps)
            df["microprice_L1"] = micro
            if "mid" in df.columns:
                df["microprice_dev_L1"] = micro - df["mid"]

    # ------------- L3: order-flow (MBO-derived) -------------
    # We keep this very defensive/universal: only use columns that actually exist.
    adds_col = _pick_first(df, ["n_add", "n_adds", "adds", "order_adds", "mbo_adds"])
    canc_col = _pick_first(df, ["n_cancel", "n_cancels", "cancels", "order_cancels", "mbo_cancels"])
    trad_col = _pick_first(df, ["n_trade", "n_trades", "trades", "trade_count", "mbo_trades"])

    # Aggressive side counts (if present)
    buy_col  = _pick_first(df, ["n_aggr_buy", "aggr_buy", "mbo_aggr_buy", "n_buy_aggr"])
    sell_col = _pick_first(df, ["n_aggr_sell", "aggr_sell", "mbo_aggr_sell", "n_sell_aggr"])

    # Sweep depth (e.g., how many levels a trade walked through)
    sweep_col = _pick_first(df, ["sweep_depth", "max_lv_traded", "max_levels_traded"])

    l3_added = []

    if adds_col is not None:
        df["mbo_adds"] = df[adds_col].astype("float64")
        l3_added.append("mbo_adds")
    if canc_col is not None:
        df["mbo_cancels"] = df[canc_col].astype("float64")
        l3_added.append("mbo_cancels")
    if trad_col is not None:
        df["mbo_trades"] = df[trad_col].astype("float64")
        l3_added.append("mbo_trades")

    if adds_col is not None and canc_col is not None:
        df["mbo_cancel_ratio"] = df[canc_col].astype("float64") / (
            df[adds_col].astype("float64") + eps
        )
        l3_added.append("mbo_cancel_ratio")

    if adds_col is not None and trad_col is not None:
        df["mbo_trade_add_ratio"] = df[trad_col].astype("float64") / (
            df[adds_col].astype("float64") + eps
        )
        l3_added.append("mbo_trade_add_ratio")

    if buy_col is not None and sell_col is not None:
        buys = df[buy_col].astype("float64")
        sells = df[sell_col].astype("float64")
        df["mbo_aggr_buy_share"] = buys / (buys + sells + eps)
        df["mbo_aggr_signed"] = buys - sells
        l3_added.extend(["mbo_aggr_buy_share", "mbo_aggr_signed"])

    if sweep_col is not None:
        df["mbo_sweep_depth"] = df[sweep_col].astype("float64")
        l3_added.append("mbo_sweep_depth")

    if bid_depth_cols or l3_added:
        print(
            f"[BOOK] [{inst_tag or ''}] derived L2 depth cols={len(bid_depth_cols)} "
            f"imb_cols={len(imb_cols)} L3_cols={len(l3_added)}",
            flush=True,
        )

    return df

# ========================= Torch core utils =========================
def to_int64_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return index.view("i8")

@torch.no_grad()
def build_anchors_ns(start_ns: int, end_ns: int, step_ns: int, dev) -> torch.Tensor:
    return torch.arange(start_ns, end_ns, step_ns, device=dev, dtype=torch.long)

def searchsorted_asof(idx_ns_t: torch.Tensor, anchors_ns_t: torch.Tensor, offset_ns: int = 0):
    targets = anchors_ns_t + int(offset_ns)
    pos = torch.searchsorted(idx_ns_t, targets, right=True) - 1
    valid = pos >= 0
    return pos, valid

def searchsorted_future(idx_ns_t: torch.Tensor, anchors_ns_t: torch.Tensor, offset_ns: int = 0):
    targets = anchors_ns_t + int(offset_ns)
    pos = torch.searchsorted(idx_ns_t, targets, right=False)
    valid = pos < idx_ns_t.numel()
    return pos, valid

def gather_windows_batch(feats_t: torch.Tensor, pos_t: torch.Tensor, win: int = 0):
    B = pos_t.shape[0]; F = feats_t.shape[1]
    if win <= 0:
        return torch.empty((B, 0, F), device=feats_t.device), torch.ones(B, dtype=torch.bool, device=feats_t.device)
    offsets = torch.arange(win, device=feats_t.device, dtype=torch.long)
    idx_g = (pos_t.view(-1,1) - (win - 1 - offsets).view(1,-1))
    ok = (idx_g >= 0); idx_g = torch.clamp(idx_g, min=0)
    flat = idx_g.reshape(-1)
    X = feats_t.index_select(0, flat).view(B, win, F)
    if not ok.all(): X = X * ok.unsqueeze(-1)
    return X, ok.all(dim=1)

def safe_r2(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> float:
    y_true = y_true.to(torch.float64); y_pred = y_pred.to(torch.float64)
    var = torch.var(y_true, unbiased=False)
    if torch.isnan(var) or var < eps:
        mae = torch.mean(torch.abs(y_true - y_pred))
        return 1.0 if float(mae) < 1e-6 else 0.0
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2) + eps
    return float(1.0 - (ss_res / ss_tot))

# ========================= Models =========================
def _downsample_time(X: torch.Tensor, max_len: int) -> torch.Tensor:
    B, T, F = X.shape
    if T <= max_len: return X
    stride = max(1, math.ceil(T / max_len))
    Xc = X.transpose(1,2)
    pool = torch.nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True)
    Y = pool(Xc)
    return Y.transpose(1,2)

class TinyHead(torch.nn.Module):
    def __init__(self, fdim: int, out_dim: int, has_cls: bool):
        super().__init__()
        self.convA = torch.nn.Conv1d(fdim, 64, kernel_size=15, padding=7)
        self.convB = torch.nn.Conv1d(fdim, 64, kernel_size=31, padding=15)
        self.act = torch.nn.ReLU()
        self.fc_reg = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim)
        )
        for m in self.fc_reg.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                torch.nn.init.zeros_(m.bias)
        self.fc_cls = None
        if has_cls:
            self.fc_cls = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, out_dim * 3)
            )
            for m in self.fc_cls.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                    torch.nn.init.zeros_(m.bias)

    def forward(self, XA, XB, max_seq_len: int):
        XA = XA.transpose(1,2); XB = XB.transpose(1,2)
        hA = self.act(self.convA(XA)).amax(dim=-1)
        hB = self.act(self.convB(XB)).amax(dim=-1)
        h = torch.cat([hA,hB], dim=1)  # [B,128]
        y_reg = self.fc_reg(h)
        logits = None
        if self.fc_cls is not None:
            logits = self.fc_cls(h).view(-1, y_reg.shape[1], 3)
        return y_reg, logits

class LSTMHead(torch.nn.Module):
    def __init__(self, fdim: int, out_dim: int, hidden_size=64, num_layers=2, dropout=0.0, max_seq_len=4096, has_cls=False):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.lstm = torch.nn.LSTM(
            input_size=fdim, hidden_size=hidden_size, num_layers=num_layers,
            dropout=(dropout if num_layers>1 else 0.0), batch_first=True
        )
        self.fc_reg = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim)
        )
        for m in self.fc_reg.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                torch.nn.init.zeros_(m.bias)
        self.fc_cls = None
        if has_cls:
            self.fc_cls = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, out_dim * 3)
            )
            for m in self.fc_cls.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                    torch.nn.init.zeros_(m.bias)

    def forward(self, XA, XB, max_seq_len: int):
        X = XB if XA.shape[1] == 0 else torch.cat([XA, XB], dim=1)
        X = _downsample_time(X, min(self.max_seq_len, int(max_seq_len)))
        out, _ = self.lstm(X)
        h_last = out[:, -1, :]
        y_reg = self.fc_reg(h_last)
        logits = None
        if self.fc_cls is not None:
            logits = self.fc_cls(h_last).view(-1, y_reg.shape[1], 3)
        return y_reg, logits

def infer_instrument_tag(instrument: Optional[str], parent_symbol: Optional[str], files: list[str]) -> str:
    if instrument:
        return re.sub(r"[^A-Za-z0-9]+", "", instrument).upper()
    if parent_symbol:
        base = parent_symbol.split(".", 1)[0]
        return re.sub(r"[^A-Za-z0-9]+", "", base).upper()
    for p in files:
        b = os.path.basename(p)
        m = re.search(r"^[a-z]+_([A-Za-z0-9]+)_\d{8}_", b)
        if m:
            return m.group(1).upper()
    return "SYMBOL"

# ========================= Config / Args =========================
@dataclass
class RunConfig:
    data_dir: str; save_dir: str; summary_csv: str; dates: List[str] | None; note: str
    epochs: int; batch_size: int; win_a: int; win_b: int; batch_anchors: int
    sample_size: int; ts_col: str; ts_unit: str; seed: int; no_amp: bool

ap = argparse.ArgumentParser()
# legacy single root (back-compat)
ap.add_argument("--data_dir", required=False, default=None)
# multi-roots
ap.add_argument("--data_roots", nargs="+", default=None,
                help="One or more root directories containing *_SLIM.parquet (e.g., /.../2024 /.../2025). Overrides --data_dir.")
ap.add_argument("--event_type", type=str, default=None,
                help="Optional prefix to filter event type (e.g., jobs, cpi, gdp, live).")
ap.add_argument("--save_dir", required=True)
ap.add_argument("--summary_csv", required=True)
ap.add_argument("--dates", nargs="*", default=None,
                help="YYYYMM month tokens (legacy monthly mode).")
ap.add_argument("--note", type=str, default="POST30-ms; LSTM (multitask)")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch_size", type=int, default=1024)
ap.add_argument("--win_a", type=int, default=0)
ap.add_argument("--win_b", type=int, default=15000)
ap.add_argument("--batch_anchors", type=int, default=256)
ap.add_argument("--sample_size", type=int, default=0, help="0 = no cap")
ap.add_argument("--ts_col", type=str, default="ts_event")
ap.add_argument("--ts_unit", type=str, default="ns", choices=["ns","us","ms","s"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no_amp", action="store_true")
ap.add_argument("--profile", choices=["hft","idt"], default=None,
                help="Required for --event_type live. Also sets default horizons and suffixes filenames.")
# Model & horizons
ap.add_argument("--horizons_ms", type=int, nargs="+", default=[30,60,120,250])
ap.add_argument("--arch", type=str, default="lstm", choices=["lstm","tiny"])
ap.add_argument("--lstm_hidden", type=int, default=64)
ap.add_argument("--lstm_layers", type=int, default=2)
ap.add_argument("--lstm_dropout", type=float, default=0.0)
ap.add_argument("--max_seq_len", type=int, default=15000)
# Anchor spacing
ap.add_argument("--step_ms", type=int, default=30, help="anchor cadence in milliseconds")
# PT time-of-day window (optional)
ap.add_argument("--restrict_to_release_window", action="store_true",
                help="Only use anchors within [release_time_pt, release_time_pt+post_window_min) in PT.")
ap.add_argument("--release_time_pt", type=str, default="05:30", help="Release time in PT (HH:MM).")
ap.add_argument("--tod_start_pt", type=str, default=None,
                help="Alias for --release_time_pt (time-of-day start, e.g. 05:30)")
ap.add_argument("--post_window_min", type=int, default=10, help="Minutes after release to include.")
# Classification (multi-task) knobs
ap.add_argument("--has_cls", action="store_true", help="Enable classification head & metrics.")
ap.add_argument("--cls_eps_bps", type=float, default=0.5, help="Dead-zone for logret target (in bps).")
ap.add_argument("--cls_eps_points", type=float, default=0.0, help="Dead-zone for points target (in points).")
ap.add_argument("--lam_reg", type=float, default=1.0)
ap.add_argument("--lam_cls", type=float, default=1.0)
ap.add_argument("--cls_ignore_zero", action="store_true", help="Ignore 0-class in CE.")
ap.add_argument("--cls_weights", type=float, nargs=3, default=None,
                help="Class weights (w_down, w_flat, w_up) for CE.")
# L2/3 book data
ap.add_argument("--book_features",choices=["off", "auto"],default="off",help=(
        "If 'auto', derive L2/L3 depth & order-flow features when *_SLIM.parquet "
        "has book / MBO summary columns (e.g., bid_px_1/bid_sz_1, add/cancel counts). "
        "Default: off."),)
ap.add_argument("--book_max_level",type=int,default=10,help="Max book level to use when deriving depth features (default 10).",)
# GPU logging
ap.add_argument("--log_gpu", action="store_true", help="(Optional) print CUDA memory periodically.")
ap.add_argument("--gpu_log_every", type=int, default=60, help="Seconds between prints (min 5s).")
# (features companions off by default; keep your current defaults)
ap.add_argument("--znorm_window_s", type=int, default=60,
                help="Seconds for rolling std used in *_zn60 features (default 60s).")
ap.add_argument("--instrument", default=None,
                help="Symbol tag to embed in filenames (e.g., ES, ZN). If omitted, inferred from --parent_symbol or file names.")
ap.add_argument("--parent_symbol", default=None,
                help="Optional parent symbol to infer instrument when --instrument not given (e.g., ES.FUT -> ES).")
# Target + scaling
ap.add_argument("--target", choices=["logret","points"], default="logret",
                help="Prediction target: logret = log(p_t+h) - log(p_t), points = p_t+h - p_t")
ap.add_argument("--y_scale", type=float, default=1e4,
                help="Multiply targets by this scalar for training/metrics. Use 1.0 to disable.")
# ===== Rolling window & daily-file support =====
ap.add_argument("--rolling_days", type=int, default=0,
                help="If >0, train on the latest N trading days found on disk (ignores --dates).")
ap.add_argument("--start_date", type=str, default=None,
                help="Inclusive start date (YYYY-MM-DD) for day-range training (ignores --dates if provided).")
ap.add_argument("--end_date", type=str, default=None,
                help="Inclusive end date (YYYY-MM-DD) for day-range training (requires --start_date).")
ap.add_argument("--accept_daily_single", action="store_true",
                help="Allow DAY (no _POST_) files (e.g., live_YYYYMMDD_ES*_L1_SLIM.parquet).")
ap.add_argument("--post_only", dest="post_only", action="store_true", default=True,
                help="Use only _POST_ files (default True). Disable with --no-post_only.")
ap.add_argument("--no-post_only", dest="post_only", action="store_false")
# NEW: exclude today (PT) to avoid growing daily parquet
ap.add_argument(
    "--exclude_today",
    action="store_true",
    help="Exclude today's YYYYMMDD (America/Los_Angeles) from training selection to avoid growing live parquet."
)
# use L12 parquet file
ap.add_argument("--prefer_l12",action="store_true",help="Prefer L12 (mbp-10) parquet files for training; fall back to L1 if not found.")

# ===== ZVOL gating (skip bad windows) =====
ap.add_argument("--zvol_gate_p90", type=float, default=2.5,
                help="If >0, skip group when zvol_p90 exceeds this threshold. (default 2.5)")
ap.add_argument("--zvol_gate_profiles", type=str, default="idt",
                help="Comma-separated profiles to apply gate to (e.g., 'idt' or 'hft,idt').")
ap.add_argument("--zvol_gate_when", choices=["pre_anchor","post_anchor"], default="pre_anchor",
                help="When to apply gate: before anchor build or after anchor build (default pre_anchor).")

args = ap.parse_args()
torch.manual_seed(args.seed)
dev = device_of_choice()

# Was --horizons_ms explicitly provided on the CLI?
user_passed_horizons = any(
    a == "--horizons_ms" or a.startswith("--horizons_ms=")
    for a in sys.argv[1:]
)

# Validate / fill profile & horizons for live
if args.event_type == "live":
    if not args.profile:
        print("[ERROR] --profile {hft,idt} is required when --event_type live", flush=True)
        sys.exit(2)

    # Only override when the user did NOT pass horizons explicitly
    if not user_passed_horizons:
        args.horizons_ms = HFT_DEFAULTS if args.profile == "hft" else IDT_DEFAULTS

    # Daily LIVE files have no _POST_; prefer them automatically
    args.post_only = False

# ---- alias normalization ----
if args.tod_start_pt and not args.release_time_pt:
    args.release_time_pt = args.tod_start_pt

HORIZONS_MS = list(map(int, args.horizons_ms))
HORIZONS_NS = [h*1_000_000 for h in HORIZONS_MS]
OUT_DIM = len(HORIZONS_MS)
STEP_NS = int(args.step_ms) * 1_000_000

HAS_CLS = bool(args.has_cls)

if args.target == "logret":
    CLS_EPS = float(args.cls_eps_bps) / 1e4
    UNIT = "bps" if abs(args.y_scale - 1e4) < 1e-12 else ("dec" if abs(args.y_scale - 1.0) < 1e-12 else "scaled")
else:
    CLS_EPS = float(args.cls_eps_points)
    UNIT = "pts" if abs(args.y_scale - 1.0) < 1e-12 else "scaled-pts"

# ========================= Discover files =========================
roots = []
if args.data_roots and len(args.data_roots) > 0:
    roots = args.data_roots
elif args.data_dir:
    roots = [args.data_dir]
else:
    print("[ERROR] Provide --data_roots (one or more) or --data_dir", flush=True)
    sys.exit(2)

files = []
for r in roots:
    got = load_month_files(r, args.dates, args.event_type, prefer_l12=args.prefer_l12)
    print(f"[INFO] root={r} matched_files={len(got)} prefer_l12={args.prefer_l12}", flush=True)
    files.extend(got)
files = sorted(set(files))

INST_TAG = infer_instrument_tag(args.instrument, args.parent_symbol, files)
EVT_TAG  = (args.event_type or "event").lower()
print(f"[INFO] Inferred instrument tag = {INST_TAG}", flush=True)

# NEW: keep only this instrument (so L12 won't mix ES + ZN)
files = [p for p in files if re.search(rf"_{re.escape(INST_TAG)}[A-Za-z0-9]*_", os.path.basename(p), re.I)]
files = sorted(set(files))

INST_TAG = infer_instrument_tag(args.instrument, args.parent_symbol, files)
EVT_TAG  = (args.event_type or "event").lower()
print(f"[INFO] Inferred instrument tag = {INST_TAG}", flush=True)

# Build base save_dir
os.makedirs(args.save_dir, exist_ok=True)
# Summary CSV directory
os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
print(f"[INFO] Writing summary to: {os.path.abspath(args.summary_csv)}", flush=True)

if not files:
    header = build_train_summary_header(HORIZONS_MS, HAS_CLS)
    append_summary(args.summary_csv, header, dict(
        date="none", mode="train", n=0,
        MAE=float("nan"), MSE=float("nan"), R2=float("nan"), HIT_REG=float("nan"),
        epochs=args.epochs, batch_size=args.batch_size,
        train_stamp=pd.Timestamp.utcnow().isoformat(), note=args.note, model_tag="none"
    ))
    sys.exit(0)

# ---------- File filters ----------
def file_matches(p: str, inst: str, level: str, post_only: bool, accept_daily_single: bool) -> bool:
    b = os.path.basename(p)
    if not re.search(rf"_{re.escape(inst)}[A-Za-z0-9]*_", b, flags=re.IGNORECASE):
        return False
    if level and (f"_{level}_" not in b) and (not b.endswith(f"_{level}_SLIM.parquet")):
        return False
    if post_only:
        return "_POST_" in b
    else:
        if "_POST_" in b:
            return True
        # Daily "live_YYYYMMDD_*" files have no _POST_; allow when accept_daily_single
        return accept_daily_single

def extract_day_token(p: str) -> Optional[str]:
    ymd = yyyymmdd_from_path(p)
    return ymd

# Build index: day_token -> [files]
LEVEL_TAG = "L12" if getattr(args, "prefer_l12", False) else "L1"

day_map: Dict[str, List[str]] = {}
for p in files:
    if not file_matches(p, INST_TAG, LEVEL_TAG, args.post_only, args.accept_daily_single):
        continue
    d = extract_day_token(p)
    if not d:
        continue
    day_map.setdefault(d, []).append(p)

# ---- Exclude "today" (PT) if requested ----
if args.exclude_today:
    today_pt = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y%m%d")
    if today_pt in day_map:
        print(f"[INFO] Excluding today (PT) {today_pt} due to --exclude_today.", flush=True)
        day_map.pop(today_pt, None)

# Resolve rolling/day-range selection
selected_days: List[str] = []
if args.start_date:
    if not args.end_date:
        print("[ERROR] --start_date needs --end_date", flush=True); sys.exit(2)
    sd = pd.Timestamp(args.start_date).strftime("%Y%m%d")
    ed = pd.Timestamp(args.end_date).strftime("%Y%m%d")
    all_days = sorted([d for d in day_map.keys() if sd <= d <= ed])
    selected_days = all_days
elif args.rolling_days and args.rolling_days > 0:
    all_days_sorted = sorted(day_map.keys())
    selected_days = all_days_sorted[-args.rolling_days:]
else:
    # legacy monthly path: keep monthly groups
    selected_days = []

# For LIVE training we want a deterministic <start>_<end> window folder and end-date token
live_window_str = None
live_end_ymd = None
if selected_days:
    live_window_str = f"{selected_days[0]}_{selected_days[-1]}"   # underscore format
    live_end_ymd = selected_days[-1]
    live_save_root = os.path.join(args.save_dir, live_window_str)
    os.makedirs(live_save_root, exist_ok=True)
else:
    live_save_root = args.save_dir

# ========================= Build groups =========================
groups: Dict[str, List[str]] = {}

# pick which “level” we are training on
LEVEL_TAG = "L12" if getattr(args, "prefer_l12", False) else "L1"

if selected_days:
    use_files = []
    for d in selected_days:
        use_files.extend(day_map.get(d, []))

    # (optional but recommended) enforce level + inst here too in case day_map has mixed files
    use_files = [
        p for p in use_files
        if file_matches(p, INST_TAG, LEVEL_TAG, args.post_only, args.accept_daily_single)
    ]

    key = f"ROLLING[{selected_days[0]}..{selected_days[-1]}] ({len(selected_days)}d)"
    groups[key] = sorted(set(use_files))
    print(f"[INFO] Rolling/day-range selected days = {selected_days}", flush=True)
else:
    # monthly path (original behavior)
    for p in files:
        if not file_matches(p, INST_TAG, LEVEL_TAG, args.post_only, args.accept_daily_single):
            continue
        mm = yyyymm_from_path(p)
        groups.setdefault(mm, []).append(p)

print("[DEBUG] files entering group detector:")
for p in files:
    print("   ", os.path.basename(p), flush=True)
print(f"[INFO] groups_detected={list(groups.keys())}", flush=True)
print(f"[INFO] grouping_level={LEVEL_TAG} prefer_l12={getattr(args, 'prefer_l12', False)}", flush=True)

# ========================= Main group loop =========================
header = build_train_summary_header(HORIZONS_MS, HAS_CLS)

for gkey in sorted(groups.keys()):
    flist = sorted(set(groups[gkey]))
    print(f"[{gkey}] using {len(flist)} files:", flush=True)
    for p in flist:
        print(f"[{gkey}]   {p}", flush=True)

    # Hygiene reset
    torch.cuda.empty_cache()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dfs = []
    total_rows = 0
    t0m = time.time()

    if not flist:
        print(f"[{gkey}] WARN: no files match selection; skipping", flush=True)
        append_summary(args.summary_csv, header, dict(
            date=gkey, mode="train", n=0,
            MAE=float("nan"), MSE=float("nan"), R2=float("nan"), HIT_REG=float("nan"),
            epochs=args.epochs, batch_size=args.batch_size,
            train_stamp=pd.Timestamp.utcnow().isoformat(), note=args.note, model_tag="none"
        ))
        continue

    # SAFETY: skip files that are still growing or missing metadata (parquet footer not finalized)
    try:
        import pyarrow.parquet as pq
    except Exception:
        pq = None

    for i, path in enumerate(flist, 1):
        try:
            # If "today" leaked in (e.g., user didn't pass --exclude_today), this protects us:
            try:
                sz1 = os.path.getsize(path)
                time.sleep(0.8)
                sz2 = os.path.getsize(path)
                if sz2 != sz1:
                    print(f"[{gkey}] SKIP (growing): {os.path.basename(path)}", flush=True)
                    continue
            except FileNotFoundError:
                print(f"[{gkey}] SKIP (disappeared): {os.path.basename(path)}", flush=True)
                continue

            if pq is not None:
                try:
                    _ = pq.ParquetFile(path).metadata
                except Exception as meta_e:
                    print(f"[{gkey}] SKIP (no metadata yet): {os.path.basename(path)} :: {meta_e}", flush=True)
                    continue

            df = read_ticks_df(path, ts_col=args.ts_col, ts_unit=args.ts_unit)
        except Exception as e:
            print(f"[{gkey}] ERROR reading {os.path.basename(path)}: {e}", flush=True)
            continue

        if "mid" not in df.columns:
            pref = [c for c in ("mid_px","midprice","mid_price") if c in df.columns]
            if pref: df["mid"] = df[pref[0]]
            elif {"best_bid","best_ask"}.issubset(df.columns):
                df["mid"] = (df["best_bid"] + df["best_ask"]) / 2.0
            else:
                num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                            if c.lower() not in {"ts_event","timestamp","epoch","seq"}]
                if not num_cols:
                    print(f"[{gkey}] cannot synthesize 'mid' in {os.path.basename(path)}; skip", flush=True); continue
                df["mid"] = df[num_cols[0]]

        rows = len(df); total_rows += rows
        dfs.append(df)

    if not dfs:
        print(f"[{gkey}] WARN: selected files failed to read; skipping", flush=True)
        append_summary(args.summary_csv, header, dict(
            date=gkey, mode="train", n=0,
            MAE=float("nan"), MSE=float("nan"), R2=float("nan"), HIT_REG=float("nan"),
            epochs=args.epochs, batch_size=args.batch_size,
            train_stamp=pd.Timestamp.utcnow().isoformat(), note=args.note, model_tag="none"
        ))
        continue

    feats_df = pd.concat(dfs, axis=0).sort_index()
    if feats_df.index.has_duplicates:
        feats_df = feats_df[~feats_df.index.duplicated(keep="last")]
    print(f"[{gkey}] concat+dedup rows={len(feats_df)} (load_time={time.time()-t0m:.2f}s)", flush=True)

    # --- z-vol for gating/feats ---
    mid_clip = feats_df["mid"].copy()
    mid_clip = mid_clip.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0).clip(lower=1e-9)

    ret_1s   = np.log(mid_clip).diff().resample("1s").sum().fillna(0.0)
    sigma_s  = int(args.znorm_window_s)
    sigma_1s = ret_1s.rolling(f"{sigma_s}s", min_periods=max(1, sigma_s//4)).std()

    eps = 1e-12
    med1h_1s = sigma_1s.resample("1h").median()
    zvol_1s  = (sigma_1s / (med1h_1s + eps)).fillna(0.0)

    sigma_tick = sigma_1s.reindex(feats_df.index, method="ffill").bfill().fillna(0.0)
    zvol_tick  = zvol_1s.reindex(feats_df.index, method="ffill").fillna(0.0)

    # --- z-vol gating (skip bad windows) ---
    def _should_gate() -> bool:
        if args.zvol_gate_p90 <= 0:
            return False
        when_ok = (args.zvol_gate_when == "always") or (args.zvol_gate_when == "live_only" and args.event_type == "live")
        if not when_ok:
            return False

        profs = (args.zvol_gate_profiles or "").strip().lower()
        if profs == "all":
            return True
        prof_set = {p.strip() for p in profs.split(",") if p.strip()}
        # If no profile provided, don't gate (or choose to gate anyway—your call)
        if not args.profile:
            return False
        return args.profile.lower() in prof_set

    if _should_gate():
        zarr = zvol_tick.to_numpy(dtype=np.float64)
        zarr = zarr[np.isfinite(zarr)]
        if zarr.size > 0:
            z_p50 = float(np.percentile(zarr, 50))
            z_p90 = float(np.percentile(zarr, 90))
            z_p99 = float(np.percentile(zarr, 99))
        else:
            z_p50 = z_p90 = z_p99 = float("nan")

        print(f"[{gkey}] [ZVOL] p50={z_p50:.2f} p90={z_p90:.2f} p99={z_p99:.2f} gate_p90={args.zvol_gate_p90:.2f}", flush=True)

        if np.isfinite(z_p90) and (z_p90 > float(args.zvol_gate_p90)):
            print(f"[{gkey}] [ZVOL] SKIP window due to z_p90={z_p90:.2f} > {args.zvol_gate_p90:.2f}", flush=True)

            # Write a summary row so you can see it was intentionally skipped
            append_summary(args.summary_csv, header, dict(
                date=gkey, mode="skip_zvol", n=0,
                MAE=float("nan"), MSE=float("nan"), R2=float("nan"), HIT_REG=float("nan"),
                epochs=args.epochs, batch_size=args.batch_size,
                train_stamp=pd.Timestamp.utcnow().isoformat(),
                note=f"{args.note} | SKIP_ZVOL p90={z_p90:.2f} p99={z_p99:.2f} gate={args.zvol_gate_p90:.2f}",
                model_tag="none"
            ))

            # Cleanup and move to next group
            del dfs, feats_df
            gc.collect()
            continue
    # --- Optional L2/L3 book features (depth + order-flow) ---
    if args.book_features == "auto":
        feats_df = augment_with_book_features(
            feats_df,
            max_level=int(args.book_max_level),
            inst_tag=INST_TAG,
        )

    # --- Feature matrix ---
    use_cols = [c for c in feats_df.columns if c != "mid" and pd.api.types.is_numeric_dtype(feats_df[c])]
    if not use_cols:
        feats_df["mid_ret"] = feats_df["mid"].pct_change().fillna(0.0)
        use_cols = ["mid_ret"]

    feats_df["mid"] = (
        feats_df["mid"].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0).clip(lower=1e-9)
    )
    feats_df[use_cols] = (
        feats_df[use_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    )

    feat_names = list(use_cols)
    feat_hash = int(hashlib.blake2b("||".join(feat_names).encode(), digest_size=8).hexdigest(), 16)

    X_np = feats_df[feat_names].to_numpy(dtype=np.float64)
    mu   = np.nanmean(X_np, axis=0)
    sd   = np.nanstd (X_np, axis=0)
    sd   = np.where(sd < 1e-12, 1.0, sd)

    def _sig(v, k=5):
        arr = np.asarray(v, dtype=float)
        k = min(k, arr.shape[0])
        return ", ".join(f"{arr[i]:.6g}" for i in range(k))
    print(f"[{gkey}] [{INST_TAG}] norm_mu[:5]=[{_sig(mu)}]  norm_sd[:5]=[{_sig(sd)}]  fdim={len(feat_names)}", flush=True)

    if "ZN" in INST_TAG and np.nanmean(mu[:2]) > 1000:
        print(f"[{gkey}] [WARN] ZN mean too high ({np.nanmean(mu[:2]):.2f}) — normalization may be using ES scale!", flush=True)
    if "ES" in INST_TAG and np.nanmean(mu[:2]) < 100:
        print(f"[{gkey}] [WARN] ES mean unusually low ({np.nanmean(mu[:2]):.2f}) — possible wrong symbol data!", flush=True)

    X_np = ((X_np - mu) / sd).astype(np.float32)
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    assert X_np.shape[1] == len(feat_names), "X_np columns != feat_names length"

    idx_ns_np = feats_df.index.asi8
    idx_t = torch.from_numpy(idx_ns_np).to(dev)

    mid_t = torch.from_numpy(feats_df["mid"].astype("float32").to_numpy()).to(dev)
    mid_t = torch.clamp(mid_t, min=1e-9)

    X_t = torch.from_numpy(X_np).to(dev)
    nf = ~torch.isfinite(X_t)
    if nf.any(): X_t[nf] = 0.0

    T, F = X_t.shape

    # ---------- Anchors ----------
    start_ns = int(idx_ns_np.min()); end_ns = int(idx_ns_np.max())
    anchors  = build_anchors_ns(start_ns, end_ns, STEP_NS, dev=dev)
    if args.restrict_to_release_window and anchors.numel() > 0:
        anchors_dt = pd.to_datetime(anchors.detach().cpu().numpy(), unit="ns", utc=True)
        anchors_dt_pt = anchors_dt.tz_convert("America/Los_Angeles")
        hh, mm_pt = map(int, args.release_time_pt.split(":"))
        day_floor = anchors_dt_pt.floor("D")
        release_start = day_floor + pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm_pt, unit="m")
        release_end   = release_start + pd.to_timedelta(int(args.post_window_min), unit="m")
        mask_window = (anchors_dt_pt >= release_start) & (anchors_dt_pt < release_end)
        mask_np = np.asarray(mask_window, dtype=np.bool_)
        kept = int(mask_np.sum())
        anchors = anchors[torch.from_numpy(mask_np).to(anchors.device)]
        print(f"[{gkey}] window PT {args.release_time_pt}+{args.post_window_min}m: kept {kept}/{mask_np.size} anchors", flush=True)

    pos0, v0 = searchsorted_asof(idx_t, anchors, 0)
    posH_list, vH_list = [], []
    for h_ns in HORIZONS_NS:
        pH, vH = searchsorted_future(idx_t, anchors, h_ns)
        posH_list.append(pH); vH_list.append(vH)
    valid = v0.clone()
    for vH in vH_list: valid = valid & vH
    mask_full = valid & (pos0 >= (args.win_b - 1))
    for pH in posH_list: mask_full = mask_full & (pH > pos0)
    pos0 = pos0[mask_full]; posH_list = [pH[mask_full] for pH in posH_list]
    if args.sample_size > 0 and pos0.numel() > args.sample_size:
        pos0 = pos0[:args.sample_size]; posH_list = [pH[:args.sample_size] for pH in posH_list]
    print(f"[{gkey}] anchors usable={pos0.numel()}", flush=True)
    if pos0.numel() == 0:
        append_summary(args.summary_csv, header, dict(
            date=gkey, mode="train", n=0,
            MAE=float("nan"), MSE=float("nan"), R2=float("nan"), HIT_REG=float("nan"),
            epochs=args.epochs, batch_size=args.batch_size,
            train_stamp=pd.Timestamp.utcnow().isoformat(), note=args.note, model_tag="none"
        ))
        del dfs, feats_df, X_np, X_t, mid_t, idx_t, anchors
        gc.collect()
        continue

    # ---------- Model ----------
    if args.arch == "lstm":
        model = LSTMHead(F, OUT_DIM, hidden_size=args.lstm_hidden, num_layers=args.lstm_layers,
                         dropout=args.lstm_dropout, max_seq_len=args.max_seq_len, has_cls=HAS_CLS).to(dev).train()
    else:
        model = TinyHead(F, OUT_DIM, has_cls=HAS_CLS).to(dev).train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(dev.type == "cuda") and (not args.no_amp))
    use_amp = (dev.type == "cuda") and (not args.no_amp)

    loss_reg = torch.nn.SmoothL1Loss(beta=1.0)
    if HAS_CLS:
        if args.cls_weights is not None:
            w = torch.tensor(args.cls_weights, dtype=torch.float32, device=dev)
            ce_loss = torch.nn.CrossEntropyLoss(weight=w, reduction="mean")
        else:
            ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    rng = np.random.default_rng(args.seed)
    RESERVOIR_N = 500_000
    sample_abs = np.empty(RESERVOIR_N, dtype=np.float32); sp = 0
    n_total = 0
    min_seen = float("inf")
    max_seen = float("-inf")
    suspect_cap = 4.0 if args.target == "logret" else None
    n_at_suspect = 0
    TOL = 1e-6

    # ---------- Train ----------
    B_ANC = args.batch_anchors
    printed_y_dbg = False

    for ep in range(1, args.epochs + 1):
        t0e = time.time()
        sum_abs_s, sum_sq_s, n_err = 0.0, 0.0, 0

        for i in range(0, pos0.numel(), B_ANC):
            p0 = pos0[i:i+B_ANC]
            pHs = [pH[i:i+B_ANC] for pH in posH_list]

            ok = (p0 >= (args.win_b - 1))
            for k in range(len(pHs)): ok = ok & (pHs[k] > p0)
            if not ok.any(): continue
            p0 = p0[ok]; pHs = [pH[ok] for pH in pHs]
            if p0.numel() == 0: continue

            with torch.no_grad():
                if args.target == "logret":
                    log_p0 = torch.log(mid_t[p0])
                    Ys = [torch.log(mid_t[pH]) - log_p0 for pH in pHs]
                else:
                    p0v = mid_t[p0].to(torch.float64)
                    Ys = [(mid_t[pH].to(torch.float64) - p0v) for pH in pHs]

                Y_true = torch.stack(Ys, dim=1)
                Y_s    = torch.nan_to_num(Y_true * args.y_scale)

                if HAS_CLS:
                    true_dir = torch.where(Y_true > CLS_EPS, 1, torch.where(Y_true < -CLS_EPS, -1, 0))
                    true_cls = (true_dir + 1).to(torch.long)

                if not printed_y_dbg:
                    printed_y_dbg = True
                    absY = torch.abs(Y_true).flatten()
                    p50 = float(torch.quantile(absY, 0.50)) if absY.numel() else float("nan")
                    p90 = float(torch.quantile(absY, 0.90)) if absY.numel() else float("nan")
                    p99 = float(torch.quantile(absY, 0.99)) if absY.numel() else float("nan")
                    print(f"[{gkey}] |target| p50={p50:.3e} p90={p90:.3e} p99={p99:.3e} ({args.target})", flush=True)

            absY_all = torch.abs(Y_true).flatten().to(torch.float64)
            if absY_all.numel():
                n = int(absY_all.numel())
                n_total += n
                bmin = float(torch.min(absY_all))
                bmax = float(torch.max(absY_all))
                if bmin < min_seen: min_seen = bmin
                if bmax > max_seen: max_seen = bmax

                if suspect_cap is not None:
                    n_at_suspect += int(torch.sum(torch.isclose(
                        absY_all, torch.tensor(suspect_cap, device=absY_all.device, dtype=absY_all.dtype),
                        rtol=1e-6, atol=TOL
                    )).item())

                take = min(n, RESERVOIR_N - sp)
                if take > 0:
                    sample_abs[sp:sp+take] = absY_all[:take].cpu().numpy()
                    sp += take

            XA_b, okA = gather_windows_batch(X_t, p0, args.win_a)
            XB_b, okB = gather_windows_batch(X_t, p0, args.win_b)
            ok_w = okA & okB
            if not ok_w.any(): continue
            XA_b = torch.nan_to_num(XA_b[ok_w]); XB_b = torch.nan_to_num(XB_b[ok_w]); Y_s = torch.nan_to_num(Y_s[ok_w])
            if HAS_CLS:
                true_dir = true_dir[ok_w]; true_cls = true_cls[ok_w]

            for j in range(0, XA_b.shape[0], args.batch_size):
                xa = XA_b[j:j+args.batch_size]
                xb = XB_b[j:j+args.batch_size]
                yb = Y_s[j:j+args.batch_size]
                if xa.shape[0] == 0: continue

                with autocast("cuda", enabled=use_amp):
                    pred_s, logits = model(xa, xb, args.max_seq_len)
                    pred_s = torch.nan_to_num(pred_s)

                    l_reg = torch.nn.SmoothL1Loss(beta=1.0)(pred_s.float(), yb.float())
                    if HAS_CLS and logits is not None:
                        logits = torch.nan_to_num(logits)
                        if args.cls_ignore_zero:
                            mask2 = (true_dir[j:j+xa.shape[0]] != 0)
                            if mask2.any():
                                l_cls = ce_loss(
                                    logits[mask2].reshape(-1,3),
                                    true_cls[j:j+xa.shape[0]][mask2].reshape(-1)
                                )
                            else:
                                l_cls = torch.tensor(0.0, device=dev)
                        else:
                            l_cls = ce_loss(
                                logits.reshape(-1,3),
                                true_cls[j:j+xa.shape[0]].reshape(-1)
                            )
                    else:
                        l_cls = torch.tensor(0.0, device=dev)

                    loss = args.lam_reg * l_reg + args.lam_cls * l_cls

                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    err_s = (pred_s.to(torch.float64) - yb.to(torch.float64))
                    sum_abs_s += float(err_s.abs().sum())
                    sum_sq_s  += float((err_s**2).sum())
                    n_err     += int(err_s.numel())

        epoch_mae = (sum_abs_s / n_err) if n_err > 0 else float("nan")
        epoch_mse = (sum_sq_s  / n_err) if n_err > 0 else float("nan")
        epoch_rmse = math.sqrt(epoch_mse) if epoch_mse == epoch_mse else float("nan")
        print(f"[{gkey}] epoch {ep} done in {time.time()-t0e:.1f}s  MAE={epoch_mae:.4f} {UNIT}  RMSE={epoch_rmse:.4f} {UNIT}  MSE={epoch_mse:.4f} {UNIT}²", flush=True)

    # --- Label audit summary ---
    if n_total > 0 and sp > 0:
        x = sample_abs[:sp]
        q1, q50, q90, q99 = np.percentile(x, [1, 50, 90, 99])
        share_at_suspect = (n_at_suspect / n_total) if suspect_cap is not None else float("nan")
        print(
            f"[{gkey}] [Label Audit] |Y| min={min_seen:.6f} p1={q1:.6f} p50={q50:.6f} "
            f"p90={q90:.6f} p99={q99:.6f} max={max_seen:.6f}"
            + (f" | share_at_{suspect_cap:.1f}={share_at_suspect:.2%}" if suspect_cap is not None else ""),
            flush=True
        )
        if suspect_cap is not None and share_at_suspect >= 0.01:
            print(f"[{gkey}] [!] Potential label cap at ~{suspect_cap:.1f}: {share_at_suspect:.2%} of samples equal that value.", flush=True)

    # ---------- Evaluate (in-group) ----------
    model.eval()
    with torch.no_grad():
        preds_h = [ [] for _ in range(OUT_DIM) ]
        true_h  = [ [] for _ in range(OUT_DIM) ]
        logits_h = [ [] for _ in range(OUT_DIM) ] if HAS_CLS else None

        for i in range(0, pos0.numel(), args.batch_anchors):
            p0 = pos0[i:i+args.batch_anchors]
            pHs = [pH[i:i+args.batch_anchors] for pH in posH_list]

            ok = (p0 >= (args.win_b - 1))
            for k in range(len(pHs)): ok = ok & (pHs[k] > p0)
            if not ok.any(): continue
            p0 = p0[ok]; pHs = [pH[ok] for pH in pHs]

            XA_b, okA = gather_windows_batch(X_t, p0)
            XB_b, okB = gather_windows_batch(X_t, p0, args.win_b)
            ok_w = okA & okB
            if not ok_w.any(): continue
            XA_b = torch.nan_to_num(XA_b[ok_w]); XB_b = torch.nan_to_num(XB_b[ok_w])

            if args.target == "logret":
                log_p0 = torch.log(mid_t[p0[ok_w]])
                Y_true_b = torch.stack([torch.log(mid_t[pH[ok_w]]) - log_p0 for pH in pHs], dim=1).to(torch.float64)
            else:
                p0v = mid_t[p0[ok_w]].to(torch.float64)
                Y_true_b = torch.stack([(mid_t[pH[ok_w]].to(torch.float64) - p0v) for pH in pHs], dim=1)

            local_reg, local_logits = [], []
            for j in range(0, XA_b.shape[0], args.batch_size):
                y_reg_s, logits = model(XA_b[j:j+args.batch_size], XB_b[j:j+args.batch_size], args.max_seq_len)
                local_reg.append(torch.nan_to_num(y_reg_s.to(torch.float64)) / args.y_scale)
                if HAS_CLS and logits is not None:
                    local_logits.append(torch.nan_to_num(logits.to(torch.float64)))
            Y_pred_b = torch.cat(local_reg, dim=0)

            for k in range(OUT_DIM):
                preds_h[k].append(Y_pred_b[:, k])
                true_h[k].append(Y_true_b[:, k])
            if HAS_CLS and local_logits:
                logits_cat = torch.cat(local_logits, dim=0)
                for k in range(OUT_DIM):
                    logits_h[k].append(logits_cat[:, k, :])

        mae_h, mse_h, hit_reg_h = [float("nan")]*OUT_DIM, [float("nan")]*OUT_DIM, [float("nan")]*OUT_DIM
        all_true = []; all_pred = []

        for k in range(OUT_DIM):
            if len(preds_h[k]) == 0: continue
            ypk = torch.cat(preds_h[k]); ytk = torch.cat(true_h[k])
            finite = torch.isfinite(ypk) & torch.isfinite(ytk)
            ypk = ypk[finite]; ytk = ytk[finite]
            if ypk.numel() == 0: continue

            err_s = (ypk - ytk) * args.y_scale
            mae_h[k]  = float(err_s.abs().mean())
            mse_h[k]  = float((err_s**2).mean())
            hit_reg_h[k] = float((torch.sign(ypk) == torch.sign(ytk)).float().mean())

            all_true.append(ytk); all_pred.append(ypk)

        if len(all_true) == 0:
            n_eval = 0; mae_val=float("nan"); mse_val=float("nan"); r2=float("nan"); hit_reg_overall=float("nan")
        else:
            y_true_cat = torch.cat(all_true); y_pred_cat = torch.cat(all_pred)
            n_eval = int(y_true_cat.numel())
            err_s_cat = (y_pred_cat - y_true_cat) * args.y_scale
            mae_val = float(err_s_cat.abs().mean())
            mse_val = float((err_s_cat**2).mean())
            r2  = safe_r2(y_true_cat, y_pred_cat)
            hit_reg_overall = float((torch.sign(y_pred_cat) == torch.sign(y_true_cat)).float().mean())

        hit3_h = hit2_h = cov2_h = None
        hit3_overall = hit2_overall = cov2_overall = float("nan")
        if HAS_CLS and logits_h is not None:
            hit3_h, hit2_h, cov2_h = [float("nan")]*OUT_DIM, [float("nan")]*OUT_DIM, [float("nan")]*OUT_DIM
            all_pred_dir = []; all_true_dir = []; all_mask2 = []
            for k in range(OUT_DIM):
                if len(logits_h[k]) == 0 or len(true_h[k]) == 0: continue
                logitsk = torch.cat(logits_h[k])            # [N,3]
                probs = torch.softmax(logitsk, dim=-1)
                pred_class = probs.argmax(dim=-1)           # 0,1,2
                pred_dir = pred_class.to(torch.int64) - 1   # -> -1,0,+1

                ytk = torch.cat(true_h[k])                  # target units
                true_dir = torch.where(ytk > CLS_EPS, 1, torch.where(ytk < -CLS_EPS, -1, 0))

                hit3_h[k] = float((pred_dir == true_dir).float().mean())
                mask2 = (true_dir != 0)
                cov2_h[k] = float(mask2.float().mean()) if mask2.numel() else float("nan")
                if mask2.any():
                    hit2_h[k] = float((pred_dir[mask2] == true_dir[mask2]).float().mean())

                all_pred_dir.append(pred_dir); all_true_dir.append(true_dir); all_mask2.append(mask2)

            if all_true_dir:
                true_dir_cat = torch.cat(all_true_dir); pred_dir_cat = torch.cat(all_pred_dir); mask2_cat = torch.cat(all_mask2)
                hit3_overall = float((pred_dir_cat == true_dir_cat).float().mean())
                cov2_overall = float(mask2_cat.float().mean()) if mask2_cat.numel() else float("nan")
                if mask2_cat.any():
                    hit2_overall = float((pred_dir_cat[mask2_cat] == true_dir_cat[mask2_cat]).float().mean())

    # ---------- Save model + json ----------
    model_tag   = "lstm" if args.arch == "lstm" else "tiny"
    suffix = getattr(args, "model_suffix", "")
    # LIVE naming: model_live_<ENDDATE>_<SYMBOL>_<ARCH>_<PROFILE>.* inside <start>_<end>/ subfolder
    if args.event_type == "live" and live_window_str and live_end_ymd and args.profile:
        fname_core = f"model_live_{live_end_ymd}_{INST_TAG}_{model_tag}_{args.profile}{suffix}"
        model_pt   = os.path.join(live_save_root, fname_core + ".pt")
        model_json = os.path.join(live_save_root, fname_core + ".json")
    else:
        # legacy/event naming (unchanged)
        safe_gkey = gkey.replace(' ', '_').replace(':', '-')
        fname_core = f"model_{EVT_TAG}_{safe_gkey}_{INST_TAG}_{model_tag}{suffix}"
        model_pt   = os.path.join(args.save_dir, fname_core + ".pt")
        model_json = os.path.join(args.save_dir, fname_core + ".json")

    torch.save(model.state_dict(), model_pt)

    Fdim = len(feat_names)
    assert Fdim == X_np.shape[1] == len(mu) == len(sd), "fdim / stats mismatch"

    norm_source = {
        "symbol": INST_TAG,
        "group": gkey,
        "rows": int(len(feats_df)),
        "feat_hash": feat_hash,
        "fdim": Fdim,
        "post_only": bool(args.post_only),
        "accept_daily_single": bool(args.accept_daily_single),
        "profile": (args.profile or ""),
        "live_window": (live_window_str or "")
    }

    with open(model_json, "w") as f:
        json.dump({
            "arch": args.arch,
            "fdim": int(Fdim),
            "out_dim": int(OUT_DIM),
            "horizons_ms": HORIZONS_MS,
            "target": args.target,
            "y_scale": float(args.y_scale),
            "norm_mu": [float(x) for x in mu.tolist()],
            "norm_sd": [float(x) for x in sd.tolist()],
            "norm_source": norm_source,
            "feat_names": list(feat_names),
            "win_a": int(args.win_a),
            "win_b": int(args.win_b),
            "max_seq_len": int(args.max_seq_len),
            "step_ms": int(args.step_ms),
            "restrict_to_release_window": bool(args.restrict_to_release_window),
            "release_time_pt": args.release_time_pt,
            "post_window_min": int(args.post_window_min),
            "has_cls": bool(HAS_CLS),
            "cls_eps_bps": float(args.cls_eps_bps),
            "cls_eps_points": float(args.cls_eps_points),
            "lstm_hidden": int(getattr(args, "lstm_hidden", 64)),
            "lstm_layers": int(getattr(args, "lstm_layers", 2)),
            "lstm_dropout": float(getattr(args, "lstm_dropout", 0.0)),
            "event_type": EVT_TAG,
            "instrument": INST_TAG,
            "profile": (args.profile or ""),
            "live_window": (live_window_str or ""),
            "version": "2025-10-14_rollwin"
        }, f, indent=2)

    print(f"[{gkey}] Saved: {model_pt} & {model_json}", flush=True)

    # ---------- Summary row ----------
    per_h_pairs = {}
    for k, h in enumerate(HORIZONS_MS):
        per_h_pairs[f"MAE_{h}ms"] = float(mae_h[k])
        per_h_pairs[f"MSE_{h}ms"] = float(mse_h[k])
        per_h_pairs[f"HIT_REG_{h}ms"] = float(hit_reg_h[k])

    stamp = pd.Timestamp.utcnow().isoformat()
    _med = float(np.nanmedian(zvol_tick.to_numpy()))
    row_note = (f"{args.note} | evt={EVT_TAG} inst={INST_TAG} "
                f"target={args.target} y_scale={args.y_scale:g} unit={UNIT} "
                f"| zvol_med={_med:.2f} | group={gkey} "
                f"| profile={(args.profile or 'na')} "
                f"| live_window={(live_window_str or 'na')}")

    row = {
        "date": gkey, "mode": "train",
        "n": int(n_eval), "MAE": float(mae_val), "MSE": float(mse_val), "R2": float(r2),
        "HITREG": float(hit_reg_overall) if 'hit_reg_overall' in locals() else float("nan"),
        "HIT_REG": float(hit_reg_overall) if 'hit_reg_overall' in locals() else float("nan"),
        "epochs": int(args.epochs), "batch_size": int(args.batch_size),
        "train_stamp": stamp, "note": row_note,
        "model_tag": os.path.basename(model_pt),
        **per_h_pairs
    }
    if HAS_CLS:
        row.update(HITCLS_3C=float(hit3_overall), HITCLS_2C=float(hit2_overall), COV_2C=float(cov2_overall))
        for k,h in enumerate(HORIZONS_MS):
            row[f"HITCLS_3C_{h}ms"] = float(hit3_h[k]) if hit3_h is not None else float("nan")
            row[f"HITCLS_2C_{h}ms"] = float(hit2_h[k]) if hit2_h is not None else float("nan")
            row[f"COV_2C_{h}ms"]    = float(cov2_h[k]) if cov2_h is not None else float("nan")

    append_summary(args.summary_csv, header, row)
    print(f"[{gkey}] Summary appended. (Units: {UNIT}, MSE in {UNIT}²).", flush=True)

    # Free
    del dfs, feats_df, X_np, X_t, mid_t, idx_t, anchors, pos0, posH_list
    gc.collect()

elapsed_sec = int(time.time() - t0)
elapsed = str(datetime.timedelta(seconds=elapsed_sec))
h, m, s = str(elapsed).split(':')
print(f"[INFO] Total execution time : {int(h):02d}:{int(m):02d}:{int(s):02d}", flush=True)
print("[INFO] all groups completed.", flush=True)