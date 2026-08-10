# -*- coding: utf-8 -*-

import os
import glob
import datetime
import pandas as pd
import numpy as np

# =====================
# 参数设置
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("US_EQUITY_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "daily"))
OUTPUT_DIR = os.getenv("US_EQUITY_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))

# 市场环境数据：只使用 SPY 作为 SPX / 大盘代理
MARKET_DIR = os.getenv("US_EQUITY_MARKET_DIR", os.path.join(PROJECT_ROOT, "data", "market"))
SPY_PATH = os.path.join(MARKET_DIR, "SPY.csv")

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

FAST_LEN = 12
SLOW_LEN = 26
SIGNAL_LEN = 9

LOOKBACK_YEARS = 2
MA_BUFFER = 0.005   # close < ma14 * (1 - 0.5%) 才触发 E1

# =====================
# 新增因子字段
# 注意：只新增因子计算和输出，不改变原 signal / signal2 / E1 逻辑
# =====================
FACTOR_COLS = [
    # 原 8 个健康趋势修复启动因子
    "ma_spread_5_100",
    "ma14_slope_5d",
    "above_ma55",
    "above_ma100",
    "rvol20",
    "pre_rvol_max_5d",
    "rsi_slope_5d",
    "hist_diff",

    # 均线缠绕 / 压制 / 修复识别
    "ma_entangle_cross_50d",
    "ma_cross_count_50d",
    "last_entangle_date",
    "last_entangle_days_to_signal",
    "ma_spread_all_5_240",
    "ma_spread_all_5_240_50d_mean",
    "ma_spread_all_5_240_50d_min",

    # 缠绕后成交量异动
    "entangle_base_v",
    "entangle_to_signal_vol_max_ratio",
    "entangle_to_signal_vol_burst_count_2x",
    "entangle_to_signal_vol_burst_count_3x",
    "entangle_to_signal_vol_ratios",

    # 缠绕后 RSI 修复
    "rsi_min_after_entangle",
    "rsi_has_pullback_after_entangle",
    "rsi_cross_below_ma_date_after_entangle",
    "rsi_cross_above_ma_date_after_entangle",
    "rsi_cross_above_ma_count_after_entangle",
    "rsi_days_from_cross_above_to_signal",

    # CCI 与缠绕后 CCI 修复
    "cci",
    "cci_ma",
    "cci_min_after_entangle",
    "cci_cross_below_ma_date_after_entangle",
    "cci_cross_above_ma_date_after_entangle",
    "cci_cross_above_ma_count_after_entangle",
    "cci_days_from_cross_above_to_signal",

    # MA10 slope 修复
    "ma10_slope_4d",
    "ma10_slope_turn_positive_date_after_entangle",
    "ma10_slope_turn_positive_count_after_entangle",
    "ma10_slope_days_from_positive_to_signal",

    # MACD 柱体状态与缠绕后 MACD 修复
    "hist_color",
    "hist_green_bar_count",
    "hist_red_bar_count",
    "hist_phase",
    "hist_expanding_today",
    "hist_expanding_count_5d",
    "hist_min_after_entangle",
    "hist_max_after_entangle",
    "hist_cross_above_zero_date_after_entangle",
    "hist_cross_above_zero_count_after_entangle",
    "hist_days_from_cross_above_zero_to_signal",
    "hist_expanding_days_after_entangle",

    # 前期 25 天结构与突破/跳空辅助因子
    "ma_spread_5_100_25d_mean",
    "ma_spread_5_100_25d_change",
    "close_to_ma100",
    "below_ma55_days_25",
    "below_ma100_days_25",
    "range_25d_pct",
    "ret_25d_past",
    "volatility_25d",
    "breakout_20",
    "close_near_high",
    "gap_pct",
    "turnover_value",
    "amount_ratio_20",

    # 外部市场环境：SPY 作为 SPX / 大盘代理
    "spy_close",
    "spy_ma20",
    "spy_ma50",
    "spy_ma100",
    "spy_ma200",
    "spy_above_ma20",
    "spy_above_ma50",
    "spy_above_ma100",
    "spy_above_ma200",
    "spy_ma20_slope_5d",
    "spy_ma50_slope_20d",
    "spy_ret_5d",
    "spy_ret_20d",
    "spy_regime",
    "spy_risk_score",
]

SIGNAL_INFO_COLS = [
    "date", "code", "open", "high", "low", "close", "volume",
    "signal2", "signal", "c1", "effect", "rsi", "rsi_ma",
    "macdLine", "signalLine", "hist",
    "ma5", "ma10", "ma14", "ma20", "ma30", "ma55", "ma60", "ma100", "ma120", "ma240",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================
# RSI 计算函数
# =====================
def calculate_rsi_wilder(close, length=14):
    close = pd.Series(close).astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =====================
# CCI 计算函数
# =====================
def calculate_cci(df, length=20):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_ma = typical_price.rolling(length).mean()

    mean_dev = typical_price.rolling(length).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    cci = (typical_price - tp_ma) / (0.015 * mean_dev)
    return cci


# =====================
# 连续 MACD 柱子颜色计数
# =====================
def add_hist_bar_state(df):
    df["hist_color"] = np.where(
        df["hist"] > 0,
        "green",
        np.where(df["hist"] < 0, "red", "neutral")
    )

    green_count = []
    red_count = []
    g_count = 0
    r_count = 0

    for hist_value in df["hist"]:
        if pd.isna(hist_value) or hist_value == 0:
            g_count = 0
            r_count = 0
        elif hist_value > 0:
            g_count += 1
            r_count = 0
        else:
            r_count += 1
            g_count = 0

        green_count.append(g_count)
        red_count.append(r_count)

    df["hist_green_bar_count"] = green_count
    df["hist_red_bar_count"] = red_count

    def classify_hist_phase(row):
        if row["hist"] > 0:
            n = row["hist_green_bar_count"]
            if n <= 3:
                return "green_early"
            elif n <= 8:
                return "green_mid"
            else:
                return "green_late"
        elif row["hist"] < 0:
            n = row["hist_red_bar_count"]
            if n <= 3:
                return "red_early"
            elif n <= 8:
                return "red_mid"
            else:
                return "red_late"
        else:
            return "neutral"

    df["hist_phase"] = df.apply(classify_hist_phase, axis=1)
    df["hist_expanding_today"] = (df["hist_diff"] > 0).astype(int)
    df["hist_expanding_count_5d"] = (
        (df["hist_diff"] > 0).astype(int).shift(1).rolling(5).sum()
    )

    return df


# =====================
# 新增：全部附加因子
# =====================
def add_research_factors(df):
    """
    只新增因子，不改变原 signal2 生成逻辑。
    所有 rolling + shift(1) 的字段，尽量只看信号日前状态，避免未来函数。
    """

    # ---------- 均线补充 ----------
    ma_windows = [5, 10, 14, 20, 30, 55, 60, 100, 120, 240]
    for w in ma_windows:
        col = f"ma{w}"
        if col not in df.columns:
            df[col] = df["close"].rolling(w).mean()

    # ---------- 原 8 个基础因子 ----------
    ma_cols_5_100 = ["ma5", "ma14", "ma20", "ma55", "ma100"]
    df["ma_spread_5_100"] = (
        df[ma_cols_5_100].max(axis=1) - df[ma_cols_5_100].min(axis=1)
    ) / df["close"]

    df["ma14_slope_5d"] = df["ma14"] / df["ma14"].shift(5) - 1
    df["above_ma55"] = (df["close"] > df["ma55"]).astype(int)
    df["above_ma100"] = (df["close"] > df["ma100"]).astype(int)

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["rvol20"] = df["volume"] / df["vol_ma20"]
    df["pre_rvol_max_5d"] = df["rvol20"].shift(1).rolling(5).max()

    df["rsi_slope_5d"] = df["rsi"] - df["rsi"].shift(5)
    df["hist_diff"] = df["hist"] - df["hist"].shift(1)

    # ---------- CCI ----------
    df["cci"] = calculate_cci(df, length=20)
    df["cci_ma"] = df["cci"].rolling(14).mean()

    # ---------- 25 天结构辅助 ----------
    df["ma_spread_5_100_25d_mean"] = df["ma_spread_5_100"].shift(1).rolling(25).mean()
    df["ma_spread_5_100_25d_change"] = (
        df["ma_spread_5_100"].shift(1) / df["ma_spread_5_100"].shift(25) - 1
    )
    df["close_to_ma100"] = df["close"] / df["ma100"] - 1
    df["below_ma55_days_25"] = (
        (df["close"] < df["ma55"]).astype(int).shift(1).rolling(25).sum()
    )
    df["below_ma100_days_25"] = (
        (df["close"] < df["ma100"]).astype(int).shift(1).rolling(25).sum()
    )
    df["range_25d_pct"] = (
        df["high"].shift(1).rolling(25).max()
        / df["low"].shift(1).rolling(25).min()
        - 1
    )
    df["ret_25d_past"] = df["close"].shift(1) / df["close"].shift(26) - 1
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_25d"] = df["log_return"].shift(1).rolling(25).std()

    df["high_20_prev"] = df["high"].shift(1).rolling(20).max()
    df["breakout_20"] = (df["close"] > df["high_20_prev"]).astype(int)
    day_range = df["high"] - df["low"]
    df["close_near_high"] = np.where(day_range != 0, (df["close"] - df["low"]) / day_range, np.nan)
    df["gap_pct"] = df["open"] / df["close"].shift(1) - 1

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["turnover_value"] = df["amount"]
    else:
        df["turnover_value"] = df["close"] * df["volume"]

    df["turnover_value_ma20"] = df["turnover_value"].rolling(20).mean()
    df["amount_ratio_20"] = df["turnover_value"] / df["turnover_value_ma20"]

    # ---------- 均线缠绕：短期 vs 中远期任意交叉 ----------
    short_ma_cols = ["ma5", "ma10", "ma14"]
    mid_ma_cols = ["ma20", "ma30", "ma60"]
    long_ma_cols = ["ma120", "ma240"]
    cross_base_cols = mid_ma_cols + long_ma_cols

    cross_flags = []
    for base_col in cross_base_cols:
        for short_col in short_ma_cols:
            diff_now = df[base_col] - df[short_col]
            diff_prev = diff_now.shift(1)
            cross = ((diff_now * diff_prev) < 0).astype(int)
            cross_name = f"cross_{base_col}_{short_col}"
            df[cross_name] = cross
            cross_flags.append(cross_name)

    if cross_flags:
        df["ma_cross_any"] = df[cross_flags].max(axis=1)
    else:
        df["ma_cross_any"] = 0

    df["ma_entangle_cross_50d"] = (
        df["ma_cross_any"].shift(1).rolling(50).max().fillna(0).astype(int)
    )
    df["ma_cross_count_50d"] = df["ma_cross_any"].shift(1).rolling(50).sum()

    ma_group_cols = ["ma5", "ma10", "ma14", "ma20", "ma30", "ma60", "ma120", "ma240"]
    df["ma_spread_all_5_240"] = (
        df[ma_group_cols].max(axis=1) - df[ma_group_cols].min(axis=1)
    ) / df["close"]
    df["ma_spread_all_5_240_50d_mean"] = df["ma_spread_all_5_240"].shift(1).rolling(50).mean()
    df["ma_spread_all_5_240_50d_min"] = df["ma_spread_all_5_240"].shift(1).rolling(50).min()

    # ---------- MACD 柱体信号日状态 ----------
    df = add_hist_bar_state(df)

    # ---------- MA10 slope ----------
    df["ma10_slope_4d"] = df["ma10"] / df["ma10"].shift(4) - 1

    # ---------- 缠绕后路径因子：成交量 / RSI / CCI / MA10 slope / MACD hist ----------
    path_cols_default = {
        "last_entangle_date": pd.NaT,
        "last_entangle_days_to_signal": np.nan,

        "entangle_base_v": np.nan,
        "entangle_to_signal_vol_max_ratio": np.nan,
        "entangle_to_signal_vol_burst_count_2x": 0,
        "entangle_to_signal_vol_burst_count_3x": 0,
        "entangle_to_signal_vol_ratios": "",

        "rsi_min_after_entangle": np.nan,
        "rsi_has_pullback_after_entangle": 0,
        "rsi_cross_below_ma_date_after_entangle": pd.NaT,
        "rsi_cross_above_ma_date_after_entangle": pd.NaT,
        "rsi_cross_above_ma_count_after_entangle": 0,
        "rsi_days_from_cross_above_to_signal": np.nan,

        "cci_min_after_entangle": np.nan,
        "cci_cross_below_ma_date_after_entangle": pd.NaT,
        "cci_cross_above_ma_date_after_entangle": pd.NaT,
        "cci_cross_above_ma_count_after_entangle": 0,
        "cci_days_from_cross_above_to_signal": np.nan,

        "ma10_slope_turn_positive_date_after_entangle": pd.NaT,
        "ma10_slope_turn_positive_count_after_entangle": 0,
        "ma10_slope_days_from_positive_to_signal": np.nan,

        "hist_min_after_entangle": np.nan,
        "hist_max_after_entangle": np.nan,
        "hist_cross_above_zero_date_after_entangle": pd.NaT,
        "hist_cross_above_zero_count_after_entangle": 0,
        "hist_days_from_cross_above_zero_to_signal": np.nan,
        "hist_expanding_days_after_entangle": 0,
    }

    for col, default in path_cols_default.items():
        df[col] = default

    cross_idx_list = df.index[df["ma_cross_any"] == 1].tolist()

    # 关键优化：路径型因子只对原 signal2 == 1 的信号触发行计算。
    # 其他普通 rolling / 当日状态因子仍然全量向量化计算，供信号行取值。
    # 这样不改变原 signal2 逻辑，只避免对每一个交易日都做缠绕后路径扫描。
    signal_idx_list = df.index[df["signal2"] == 1].tolist()

    for i in signal_idx_list:
        # 找信号日前 50 天内、且距离信号日至少 3 天的最近一次缠绕日
        # 至少 3 天是为了 base_V 使用 entangle_day + 2 时不偷看信号日之后的数据
        valid_cross_idx = [idx for idx in cross_idx_list if (idx >= i - 50) and (idx <= i - 3)]
        if not valid_cross_idx:
            continue

        entangle_idx = valid_cross_idx[-1]
        df.loc[df.index[i], "last_entangle_date"] = df.loc[entangle_idx, "date"]
        df.loc[df.index[i], "last_entangle_days_to_signal"] = i - entangle_idx

        # ----- 成交量 base_V：entangle t-5 到 t+2，去掉最高和最低 -----
        base_start = entangle_idx - 5
        base_end = entangle_idx + 2
        if base_start >= 0 and base_end < len(df):
            base_vols = df["volume"].iloc[base_start:base_end + 1].dropna().tolist()
            if len(base_vols) >= 6:
                base_vols_sorted = sorted(base_vols)
                trimmed_vols = base_vols_sorted[1:-1]
                base_v = np.mean(trimmed_vols) if trimmed_vols else np.nan
                if pd.notna(base_v) and base_v > 0:
                    monitor_start = entangle_idx + 3
                    monitor_end = i
                    if monitor_start <= monitor_end:
                        monitor_vols = df["volume"].iloc[monitor_start:monitor_end + 1]
                        vol_ratios = monitor_vols / base_v
                        ratio_list = vol_ratios[vol_ratios >= 2].round(2).tolist()

                        df.loc[df.index[i], "entangle_base_v"] = base_v
                        df.loc[df.index[i], "entangle_to_signal_vol_max_ratio"] = vol_ratios.max()
                        df.loc[df.index[i], "entangle_to_signal_vol_burst_count_2x"] = int((vol_ratios >= 2).sum())
                        df.loc[df.index[i], "entangle_to_signal_vol_burst_count_3x"] = int((vol_ratios >= 3).sum())
                        df.loc[df.index[i], "entangle_to_signal_vol_ratios"] = ",".join([str(x) for x in ratio_list])

        # ----- 指标监控区间：缠绕后到当前日，包含当前日 -----
        seg_start = entangle_idx + 1
        seg_end = i
        if seg_start > seg_end:
            continue

        seg = df.iloc[seg_start:seg_end + 1].copy()
        if seg.empty:
            continue

        # RSI 路径
        df.loc[df.index[i], "rsi_min_after_entangle"] = seg["rsi"].min()
        df.loc[df.index[i], "rsi_has_pullback_after_entangle"] = int(seg["rsi"].min() < 55)

        rsi_cross_below_mask = (
            (df["rsi"].shift(1) >= df["rsi_ma"].shift(1)) &
            (df["rsi"] < df["rsi_ma"])
        )
        rsi_cross_above_mask = (
            (df["rsi"].shift(1) <= df["rsi_ma"].shift(1)) &
            (df["rsi"] > df["rsi_ma"])
        )
        rsi_below_idx = [idx for idx in df.index[seg_start:seg_end + 1] if bool(rsi_cross_below_mask.loc[idx])]
        if rsi_below_idx:
            first_below = rsi_below_idx[0]
            df.loc[df.index[i], "rsi_cross_below_ma_date_after_entangle"] = df.loc[first_below, "date"]
            rsi_above_idx = [idx for idx in df.index[first_below + 1:seg_end + 1] if bool(rsi_cross_above_mask.loc[idx])]
        else:
            rsi_above_idx = [idx for idx in df.index[seg_start:seg_end + 1] if bool(rsi_cross_above_mask.loc[idx])]
        if rsi_above_idx:
            last_above = rsi_above_idx[-1]
            df.loc[df.index[i], "rsi_cross_above_ma_date_after_entangle"] = df.loc[last_above, "date"]
            df.loc[df.index[i], "rsi_cross_above_ma_count_after_entangle"] = len(rsi_above_idx)
            df.loc[df.index[i], "rsi_days_from_cross_above_to_signal"] = i - last_above

        # CCI 路径
        df.loc[df.index[i], "cci_min_after_entangle"] = seg["cci"].min()
        cci_cross_below_mask = (
            (df["cci"].shift(1) >= df["cci_ma"].shift(1)) &
            (df["cci"] < df["cci_ma"])
        )
        cci_cross_above_mask = (
            (df["cci"].shift(1) <= df["cci_ma"].shift(1)) &
            (df["cci"] > df["cci_ma"])
        )
        cci_below_idx = [idx for idx in df.index[seg_start:seg_end + 1] if bool(cci_cross_below_mask.loc[idx])]
        if cci_below_idx:
            first_below = cci_below_idx[0]
            df.loc[df.index[i], "cci_cross_below_ma_date_after_entangle"] = df.loc[first_below, "date"]
            cci_above_idx = [idx for idx in df.index[first_below + 1:seg_end + 1] if bool(cci_cross_above_mask.loc[idx])]
        else:
            cci_above_idx = [idx for idx in df.index[seg_start:seg_end + 1] if bool(cci_cross_above_mask.loc[idx])]
        if cci_above_idx:
            last_above = cci_above_idx[-1]
            df.loc[df.index[i], "cci_cross_above_ma_date_after_entangle"] = df.loc[last_above, "date"]
            df.loc[df.index[i], "cci_cross_above_ma_count_after_entangle"] = len(cci_above_idx)
            df.loc[df.index[i], "cci_days_from_cross_above_to_signal"] = i - last_above

        # MA10 slope 由负转正
        ma10_slope_turn_pos_mask = (
            (df["ma10_slope_4d"].shift(1) <= 0) &
            (df["ma10_slope_4d"] > 0)
        )
        ma10_pos_idx = [idx for idx in df.index[seg_start:seg_end + 1] if bool(ma10_slope_turn_pos_mask.loc[idx])]
        if ma10_pos_idx:
            last_pos = ma10_pos_idx[-1]
            df.loc[df.index[i], "ma10_slope_turn_positive_date_after_entangle"] = df.loc[last_pos, "date"]
            df.loc[df.index[i], "ma10_slope_turn_positive_count_after_entangle"] = len(ma10_pos_idx)
            df.loc[df.index[i], "ma10_slope_days_from_positive_to_signal"] = i - last_pos

        # MACD hist 路径
        df.loc[df.index[i], "hist_min_after_entangle"] = seg["hist"].min()
        df.loc[df.index[i], "hist_max_after_entangle"] = seg["hist"].max()

        hist_cross_above_zero_mask = (
            (df["hist"].shift(1) <= 0) &
            (df["hist"] > 0)
        )
        hist_above_idx = [idx for idx in df.index[seg_start:seg_end + 1] if bool(hist_cross_above_zero_mask.loc[idx])]
        if hist_above_idx:
            last_hist_above = hist_above_idx[-1]
            df.loc[df.index[i], "hist_cross_above_zero_date_after_entangle"] = df.loc[last_hist_above, "date"]
            df.loc[df.index[i], "hist_cross_above_zero_count_after_entangle"] = len(hist_above_idx)
            df.loc[df.index[i], "hist_days_from_cross_above_zero_to_signal"] = i - last_hist_above

        df.loc[df.index[i], "hist_expanding_days_after_entangle"] = int((seg["hist_diff"] > 0).sum())

    return df


# =====================
# 单标的信号检测
# =====================
def detect_signal_from_csv(csv_path, stock_code, ddate):
    try:
        df = pd.read_csv(csv_path)

        # 日期处理
        if "date" not in df.columns:
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
            else:
                print(f"[跳过] {stock_code} 没有 date/timestamp 字段")
                return pd.DataFrame(), pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").copy().reset_index(drop=True)

        # 基础字段检查
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[跳过] {stock_code} 缺少字段: {missing_cols}")
            return pd.DataFrame(), pd.DataFrame()

        numeric_cols = required_cols.copy()
        if "amount" in df.columns:
            numeric_cols.append("amount")

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])

        if len(df) < 80:
            print(f"[跳过] {stock_code} 数据太少: {len(df)}")
            return pd.DataFrame(), pd.DataFrame()

        # =====================
        # MACD
        # =====================
        ema_fast = df["close"].ewm(span=FAST_LEN, adjust=False).mean()
        ema_slow = df["close"].ewm(span=SLOW_LEN, adjust=False).mean()

        df["macdLine"] = ema_fast - ema_slow
        df["signalLine"] = df["macdLine"].ewm(span=SIGNAL_LEN, adjust=False).mean()
        df["hist"] = df["macdLine"] - df["signalLine"]

        # =====================
        # RSI / RSI MA / 原 MA
        # =====================
        df["rsi"] = calculate_rsi_wilder(df["close"], length=14)
        df["rsi_ma"] = df["rsi"].rolling(14).mean()

        df["ma14"] = df["close"].rolling(14).mean()
        df["ma5"] = df["close"].rolling(5).mean()

        # 补充均线，供新增因子使用；不参与原 signal2 逻辑
        for w in [10, 20, 30, 55, 60, 100, 120, 240]:
            df[f"ma{w}"] = df["close"].rolling(w).mean()

        # =====================
        # 原信号逻辑：不要改
        # =====================
        c1_list = []
        effect_list = []
        signal2_list = []
        signal_list = []

        c1 = np.nan
        effect = 0
        above70 = False
        signal_used = False
        below_ma14 = False

        for i in range(1, len(df)):
            rsi_prev = df["rsi"].iloc[i - 1]
            rsi_now = df["rsi"].iloc[i]

            open1 = df["open"].iloc[i]
            close = df["close"].iloc[i]
            close1 = df["close"].iloc[i - 1]
            high = df["high"].iloc[i]

            ma14 = df["ma14"].iloc[i]
            ma5 = df["ma5"].iloc[i]
            rsi_ma = df["rsi_ma"].iloc[i]

            if rsi_now > 70:
                above70 = True

            # 注意：这里 c1 取的是 RSI 从 >=70 跌破到 <70 前一天的收盘价
            if rsi_prev >= 70 and rsi_now < 70:
                c1 = df["close"].iloc[i - 1]
                effect = 0
                above70 = False

            if not np.isnan(c1) and effect == 0:
                if close1 < ma14:
                    effect = 1

            if close < ma14:
                below_ma14 = True

            if below_ma14 and close > ma5:
                signal_used = False
                below_ma14 = False

            signal2 = 0
            if effect == 1 and close > c1 and close > open1 and not signal_used:
                signal2 = 1
                signal_used = True

            signal = 0
            if effect == 1 and high > c1 and rsi_ma > 50 and rsi_now < rsi_ma:
                signal = 1

            c1_list.append(c1)
            effect_list.append(effect)
            signal2_list.append(signal2)
            signal_list.append(signal)

        df = df.iloc[1:].copy().reset_index(drop=True)

        df["c1"] = c1_list
        df["effect"] = effect_list
        df["signal2"] = signal2_list
        df["signal"] = signal_list
        df["code"] = stock_code

        # =====================
        # 新增全部研究因子：不参与原 signal2 生成
        # =====================
        df = add_research_factors(df)

        ddate = pd.to_datetime(ddate)

        # =====================
        # 最终信号筛选：不要改
        # 这里只保留过去两年内的信号
        # =====================
        signal_result = df[
            (df["signal2"] == 1)
            & (df["date"] >= ddate)
            & (df["hist"] > 0)
            & (df["rsi"] < 81)
        ].copy()

        return df, signal_result

    except Exception as e:
        print(f"[错误] {stock_code}: {e}")
        return pd.DataFrame(), pd.DataFrame()


# =====================
# 未来 10 天评估
# =====================
def calc_future_10d_metrics(df, signal_date, buy_price, signal_close):
    future10 = df[df["date"] > signal_date].head(10).copy()

    if future10.empty:
        return {
            "future_10d_end_date": pd.NaT,
            "future_10d_cum_return_pct": np.nan,
            "future_10d_max_single_day_gain_pct": np.nan,
            "future_10d_max_high_return_pct": np.nan,
            "future_10d_max_drawdown_pct": np.nan,
        }

    end_row = future10.iloc[-1]
    future_10d_cum_return_pct = (end_row["close"] - buy_price) / buy_price * 100

    close_series = pd.concat([
        pd.Series([signal_close]),
        future10["close"].reset_index(drop=True)
    ], ignore_index=True)
    daily_ret = close_series.pct_change().dropna()
    future_10d_max_single_day_gain_pct = daily_ret.max() * 100 if len(daily_ret) > 0 else np.nan

    future_10d_max_high_return_pct = (future10["high"].max() - buy_price) / buy_price * 100
    future_10d_max_drawdown_pct = (future10["low"].min() - buy_price) / buy_price * 100

    return {
        "future_10d_end_date": end_row["date"],
        "future_10d_cum_return_pct": future_10d_cum_return_pct,
        "future_10d_max_single_day_gain_pct": future_10d_max_single_day_gain_pct,
        "future_10d_max_high_return_pct": future_10d_max_high_return_pct,
        "future_10d_max_drawdown_pct": future_10d_max_drawdown_pct,
    }


# =====================
# E1 退出评估 + 合并信号信息 + 合并新增因子
# =====================
def evaluate_E1_exit_by_ma14(full_df, signal_df, ma_buffer=0.005):
    """
    E1 退出规则：
    信号触发后，如果某一天 close < ma14 * (1 - ma_buffer)，
    则认为 MA14 退出触发，记为 E1。

    同时输出：
    1. 信号日基础信息
    2. 新增因子
    3. E1 评估
    4. 信号触发后未来 10 天累计涨幅、最大单日涨幅、最大回撤
    """

    if full_df.empty or signal_df.empty:
        return pd.DataFrame()

    df = full_df.copy()
    sig_df = signal_df.copy()

    df["date"] = pd.to_datetime(df["date"])
    sig_df["date"] = pd.to_datetime(sig_df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    records = []

    latest_row = df.iloc[-1]
    latest_date = latest_row["date"]
    latest_close = latest_row["close"]

    for _, sig in sig_df.iterrows():
        signal_date = sig["date"]
        buy_price = sig["open"]   # 你的规则：信号日 open 买入
        signal_close = sig.get("close", np.nan)

        signal_info_values = {
            col: sig.get(col, np.nan)
            for col in SIGNAL_INFO_COLS
        }
        signal_factor_values = {
            col: sig.get(col, np.nan)
            for col in FACTOR_COLS
        }
        future_10d_values = calc_future_10d_metrics(
            df=df,
            signal_date=signal_date,
            buy_price=buy_price,
            signal_close=signal_close,
        )

        future_df = df[df["date"] > signal_date].copy()

        if future_df.empty:
            records.append({
                **signal_info_values,
                "signal_date": signal_date,
                "buy_price_open": buy_price,
                "E1_exit_date": pd.NaT,
                "E1_exit_close": np.nan,
                "E1_exit_ma14": np.nan,
                "exit_status": "no_future_data",
                "E1_exit_return_pct": np.nan,
                "current_date": latest_date,
                "current_close": latest_close,
                "current_return_pct": (latest_close - buy_price) / buy_price * 100,
                "max_high_date": pd.NaT,
                "max_high_before_E1": np.nan,
                "max_profit_before_E1_pct": np.nan,
                **future_10d_values,
                **signal_factor_values,
            })
            continue

        # 找 E1：收盘价低于 MA14 超过 0.5%
        e1_df = future_df[
            future_df["close"] < future_df["ma14"] * (1 - ma_buffer)
        ].copy()

        if len(e1_df) > 0:
            e1_row = e1_df.iloc[0]

            e1_exit_date = e1_row["date"]
            e1_exit_close = e1_row["close"]
            e1_exit_ma14 = e1_row["ma14"]
            exit_status = "E1_triggered"

            E1_exit_return_pct = (e1_exit_close - buy_price) / buy_price * 100

            # 最高浮盈统计到 E1 当天为止
            before_e1_df = future_df[future_df["date"] <= e1_exit_date].copy()

        else:
            e1_exit_date = pd.NaT
            e1_exit_close = np.nan
            e1_exit_ma14 = np.nan
            exit_status = "holding_no_E1_yet"

            # 没有退出信号，暂时按最新收盘价计算收益
            E1_exit_return_pct = (latest_close - buy_price) / buy_price * 100

            before_e1_df = future_df.copy()

        if len(before_e1_df) > 0:
            max_idx = before_e1_df["high"].idxmax()
            max_high_date = df.loc[max_idx, "date"]
            max_high = df.loc[max_idx, "high"]
            max_profit_pct = (max_high - buy_price) / buy_price * 100
        else:
            max_high_date = pd.NaT
            max_high = np.nan
            max_profit_pct = np.nan

        current_return_pct = (latest_close - buy_price) / buy_price * 100

        records.append({
            **signal_info_values,
            "signal_date": signal_date,
            "buy_price_open": buy_price,

            "E1_exit_date": e1_exit_date,
            "E1_exit_close": e1_exit_close,
            "E1_exit_ma14": e1_exit_ma14,
            "exit_status": exit_status,

            "E1_exit_return_pct": E1_exit_return_pct,

            "current_date": latest_date,
            "current_close": latest_close,
            "current_return_pct": current_return_pct,

            "max_high_date": max_high_date,
            "max_high_before_E1": max_high,
            "max_profit_before_E1_pct": max_profit_pct,

            **future_10d_values,
            **signal_factor_values,
        })

    return pd.DataFrame(records)




# =====================
# 市场环境：读取 SPY 并计算大盘状态
# =====================
def load_spy_market_features(spy_path=SPY_PATH):
    """
    使用 SPY 作为 SPX / 大盘环境代理。
    只用于给信号日 merge 外部市场状态，不参与原 signal2 生成逻辑。
    如果 SPY.csv 不存在，返回空表，主程序会继续跑。
    """
    if not os.path.exists(spy_path):
        print(f"[提示] 未找到 SPY 市场数据: {spy_path}，将跳过市场环境因子")
        return pd.DataFrame()

    try:
        spy = pd.read_csv(spy_path)

        if "date" not in spy.columns:
            if "timestamp" in spy.columns:
                spy["date"] = pd.to_datetime(spy["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                print(f"[提示] SPY 数据没有 date/timestamp 字段，将跳过市场环境因子")
                return pd.DataFrame()

        spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
        spy = spy.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        required_cols = ["open", "high", "low", "close"]
        missing_cols = [c for c in required_cols if c not in spy.columns]
        if missing_cols:
            print(f"[提示] SPY 数据缺少字段 {missing_cols}，将跳过市场环境因子")
            return pd.DataFrame()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in spy.columns:
                spy[col] = pd.to_numeric(spy[col], errors="coerce")

        spy["spy_close"] = spy["close"]
        spy["spy_ma20"] = spy["close"].rolling(20).mean()
        spy["spy_ma50"] = spy["close"].rolling(50).mean()
        spy["spy_ma100"] = spy["close"].rolling(100).mean()
        spy["spy_ma200"] = spy["close"].rolling(200).mean()

        spy["spy_above_ma20"] = (spy["close"] > spy["spy_ma20"]).astype(int)
        spy["spy_above_ma50"] = (spy["close"] > spy["spy_ma50"]).astype(int)
        spy["spy_above_ma100"] = (spy["close"] > spy["spy_ma100"]).astype(int)
        spy["spy_above_ma200"] = (spy["close"] > spy["spy_ma200"]).astype(int)

        spy["spy_ma20_slope_5d"] = spy["spy_ma20"] / spy["spy_ma20"].shift(5) - 1
        spy["spy_ma50_slope_20d"] = spy["spy_ma50"] / spy["spy_ma50"].shift(20) - 1
        spy["spy_ret_5d"] = spy["close"] / spy["close"].shift(5) - 1
        spy["spy_ret_20d"] = spy["close"] / spy["close"].shift(20) - 1

        def classify_spy_regime(row):
            if pd.isna(row.get("spy_close")) or pd.isna(row.get("spy_ma200")):
                return "unknown"

            # panic / risk_off：大盘跌破长期或中期均线
            if row["spy_above_ma200"] == 0:
                return "panic"

            if row["spy_above_ma50"] == 0:
                return "risk_off"

            # risk_on：站上 20/50/200，且短中期均线斜率为正
            if (
                row["spy_above_ma20"] == 1
                and row["spy_above_ma50"] == 1
                and row["spy_above_ma200"] == 1
                and row["spy_ma20_slope_5d"] > 0
                and row["spy_ma50_slope_20d"] > 0
            ):
                return "risk_on"

            return "neutral"

        spy["spy_regime"] = spy.apply(classify_spy_regime, axis=1)

        # 分数越高，外部市场越友好。只基于 SPY，不用 VIX。
        spy["spy_risk_score"] = (
            spy["spy_above_ma20"].fillna(0)
            + spy["spy_above_ma50"].fillna(0)
            + spy["spy_above_ma100"].fillna(0)
            + spy["spy_above_ma200"].fillna(0)
            + (spy["spy_ma20_slope_5d"] > 0).astype(int)
            + (spy["spy_ma50_slope_20d"] > 0).astype(int)
        )

        keep_cols = [
            "date",
            "spy_close",
            "spy_ma20",
            "spy_ma50",
            "spy_ma100",
            "spy_ma200",
            "spy_above_ma20",
            "spy_above_ma50",
            "spy_above_ma100",
            "spy_above_ma200",
            "spy_ma20_slope_5d",
            "spy_ma50_slope_20d",
            "spy_ret_5d",
            "spy_ret_20d",
            "spy_regime",
            "spy_risk_score",
        ]

        market_df = spy[keep_cols].copy()
        print(f"SPY 市场环境数据已加载: {spy_path}, 行数: {len(market_df)}")
        return market_df

    except Exception as e:
        print(f"[提示] 读取 SPY 市场数据失败: {e}，将跳过市场环境因子")
        return pd.DataFrame()


def merge_market_features(all_eval, market_df):
    """
    将 SPY 市场环境按 signal_date 合并到最终合并表。
    只做输出增强，不改变任何信号 / E1 逻辑。
    """
    if all_eval.empty:
        return all_eval

    if market_df is None or market_df.empty:
        # 保证输出字段存在，便于后续 Excel 分析
        for col in [
            "spy_close", "spy_ma20", "spy_ma50", "spy_ma100", "spy_ma200",
            "spy_above_ma20", "spy_above_ma50", "spy_above_ma100", "spy_above_ma200",
            "spy_ma20_slope_5d", "spy_ma50_slope_20d", "spy_ret_5d", "spy_ret_20d",
            "spy_regime", "spy_risk_score"
        ]:
            if col not in all_eval.columns:
                all_eval[col] = np.nan
        all_eval["spy_regime"] = all_eval["spy_regime"].fillna("unknown")
        return all_eval

    out = all_eval.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    mkt = market_df.copy()
    mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce")

    out = pd.merge(
        out,
        mkt,
        left_on="signal_date",
        right_on="date",
        how="left",
        suffixes=("", "_spy_market")
    )

    if "date_spy_market" in out.columns:
        out = out.drop(columns=["date_spy_market"])
    elif "date_y" in out.columns:
        out = out.drop(columns=["date_y"])

    return out


# =====================
# 主程序：扫描全市场 CSV
# =====================
def main():
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=LOOKBACK_YEARS * 365)

    print("=" * 80)
    print("开始扫描全美股日线数据", flush=True)
    print(f"数据目录: {DATA_DIR}")
    print(f"信号开始日期: {start_date}")
    print(f"MA14 E1 buffer: {MA_BUFFER}")
    print(f"SPY 市场环境路径: {SPY_PATH}")
    print("=" * 80)

    # 加载 SPY 市场环境；如果没有 SPY.csv，主流程仍然继续
    market_df = load_spy_market_features(SPY_PATH)

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    print(f"共发现 CSV 文件数量: {len(csv_files)}", flush=True)

    all_eval_list = []

    for idx, csv_path in enumerate(csv_files, start=1):
        filename = os.path.basename(csv_path)
        stock_code = filename.replace(".csv", "")

        if idx % 100 == 0:
            print(f"已处理 {idx}/{len(csv_files)}", flush=True)

        full_df, signal_df = detect_signal_from_csv(
            csv_path=csv_path,
            stock_code=stock_code,
            ddate=start_date
        )

        if signal_df.empty:
            continue

        # 做 E1 评估，并把信号信息 + 新增因子 + 未来10日评估合并在一张表
        eval_df = evaluate_E1_exit_by_ma14(
            full_df=full_df,
            signal_df=signal_df,
            ma_buffer=MA_BUFFER
        )

        if not eval_df.empty:
            all_eval_list.append(eval_df)

    # =====================
    # 汇总输出：只生成一个合并表
    # =====================
    run_date = today.strftime("%Y%m%d")
    merged_output_path = os.path.join(
        OUTPUT_DIR,
        f"all_us_signal_E1_full_factors_SPY_merged_{run_date}.csv"
    )

    if all_eval_list:
        all_eval = pd.concat(all_eval_list, ignore_index=True)

        # 合并 SPY 外部市场环境因子；只增强输出，不改变信号/E1逻辑
        all_eval = merge_market_features(all_eval, market_df)

        all_eval = all_eval.sort_values(["signal_date", "code"]).reset_index(drop=True)
        all_eval.to_csv(merged_output_path, index=False, encoding="utf-8-sig")

        print(f"合并结果已保存: {merged_output_path}")
        print(f"合并记录总数: {len(all_eval)}")

        print("\nE1 收益统计：")
        print(all_eval["E1_exit_return_pct"].describe())

        win_rate = (all_eval["E1_exit_return_pct"] > 0).mean() * 100
        print(f"E1 胜率: {win_rate:.2f}%")

        avg_return = all_eval["E1_exit_return_pct"].mean()
        print(f"E1 平均收益率: {avg_return:.2f}%")

        median_return = all_eval["E1_exit_return_pct"].median()
        print(f"E1 中位数收益率: {median_return:.2f}%")

        print("\n未来10日累计收益统计：")
        print(all_eval["future_10d_cum_return_pct"].describe())

    else:
        print("没有 E1 / 合并评估结果")

    print("=" * 80)
    print("运行完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
