

# coding: utf-8



import os

import time

import shutil

import requests

import pandas as pd

from datetime import datetime





# ============================================================

# 路径配置

# ============================================================



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.getenv("US_EQUITY_UNIVERSE_DIR", os.path.join(PROJECT_ROOT, "output", "universe"))

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "us_stock_basic_with_market_cap.csv")

BACKUP_DIR = os.path.join(OUTPUT_DIR, "backup")



os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(BACKUP_DIR, exist_ok=True)





# ============================================================

# Polygon 配置

# ============================================================



POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")



if not POLYGON_API_KEY:

    raise ValueError(

        "没有找到 POLYGON_API_KEY。请先执行：\n"

        "export POLYGON_API_KEY='你的polygon api key'"

    )



REQUEST_SLEEP = 0.30

MAX_RETRY = 3

TIMEOUT = 30





# ============================================================

# 请求函数

# ============================================================



def request_json(url, params=None, max_retry=MAX_RETRY):

    last_error = None



    for attempt in range(1, max_retry + 1):

        try:

            res = requests.get(url, params=params, timeout=TIMEOUT)



            if res.status_code == 429:

                wait_sec = 5 * attempt

                print(f"[RATE LIMIT] 429，等待 {wait_sec}s 后重试...")

                time.sleep(wait_sec)

                continue



            if res.status_code >= 500:

                wait_sec = 3 * attempt

                print(f"[SERVER ERROR] {res.status_code}，等待 {wait_sec}s 后重试...")

                time.sleep(wait_sec)

                continue



            data = res.json()

            return data



        except Exception as e:

            last_error = e

            wait_sec = 3 * attempt

            print(f"[REQUEST ERROR] attempt={attempt}, error={e}, 等待 {wait_sec}s")

            time.sleep(wait_sec)



    print(f"[FAILED] url={url}, last_error={last_error}")

    return {}





# ============================================================

# 拉取全部美股 ticker

# ============================================================



def download_all_tickers():

    url = "https://api.polygon.io/v3/reference/tickers"



    params = {

        "market": "stocks",

        "active": "true",

        "limit": 1000,

        "sort": "ticker",

        "order": "asc",

        "apiKey": POLYGON_API_KEY,

    }



    all_results = []

    page = 1



    while url:

        print(f"[TICKERS] 正在拉取第 {page} 页...")



        data = request_json(url, params=params)



        results = data.get("results", [])



        if not results:

            print("[TICKERS] 没有 results，返回内容：")

            print(data)

            break



        all_results.extend(results)



        next_url = data.get("next_url")



        if next_url:

            url = next_url

            params = {

                "apiKey": POLYGON_API_KEY,

            }

            page += 1

            time.sleep(REQUEST_SLEEP)

        else:

            break



    df = pd.DataFrame(all_results)



    print(f"[TICKERS] 原始 ticker 数量: {len(df)}")



    if df.empty:

        raise RuntimeError("ticker 列表为空，停止。")



    return df





# ============================================================

# 拉取单个 ticker 详情

# ============================================================



def download_ticker_details(ticker):

    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"



    params = {

        "apiKey": POLYGON_API_KEY,

    }



    data = request_json(url, params=params)



    results = data.get("results")



    if not results:

        print(f"[DETAIL EMPTY] {ticker} 没有 results")

        return {}



    return results





# ============================================================

# 备份旧文件

# ============================================================



def backup_old_file():

    if not os.path.exists(OUTPUT_FILE):

        return



    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_file = os.path.join(

        BACKUP_DIR,

        f"us_stock_basic_with_market_cap_{ts}.csv"

    )



    shutil.copy2(OUTPUT_FILE, backup_file)

    print(f"[BACKUP] 已备份旧文件: {backup_file}")





# ============================================================

# 主流程

# ============================================================



def main():

    print("=" * 100)

    print("[START] 更新 us_stock_basic_with_market_cap.csv")

    print("时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("输出文件:", OUTPUT_FILE)

    print("=" * 100)



    ticker_df = download_all_tickers()



    required_cols = ["ticker", "type", "active"]



    for col in required_cols:

        if col not in ticker_df.columns:

            raise ValueError(f"ticker 列表缺少字段: {col}, 当前字段: {list(ticker_df.columns)}")



    stock_pool = ticker_df[

        (ticker_df["type"] == "CS") &

        (ticker_df["active"] == True)

    ].copy()



    stock_pool = stock_pool.reset_index(drop=True)



    print(f"[FILTER] 普通股 active CS 数量: {len(stock_pool)}")



    details_list = []



    for i, ticker in enumerate(stock_pool["ticker"], start=1):

        print(f"[{i}/{len(stock_pool)}] 拉取详情: {ticker}")



        d = download_ticker_details(ticker)



        market_cap = d.get("market_cap")



        details_list.append({

            "ticker": ticker,

            "market_cap": market_cap,

            "market_cap_billion": market_cap / 1e9 if market_cap else None,

            "sic_code": d.get("sic_code"),

            "sic_description": d.get("sic_description"),

            "total_employees": d.get("total_employees"),

            "list_date": d.get("list_date"),

            "homepage_url": d.get("homepage_url"),

            "share_class_shares_outstanding": d.get("share_class_shares_outstanding"),

            "weighted_shares_outstanding": d.get("weighted_shares_outstanding"),

            "description": d.get("description"),

        })



        time.sleep(REQUEST_SLEEP)



    details_df = pd.DataFrame(details_list)



    final_df = stock_pool.merge(

        details_df,

        on="ticker",

        how="left"

    )



    final_df = final_df.sort_values("ticker").reset_index(drop=True)



    backup_old_file()



    tmp_file = OUTPUT_FILE + ".tmp"



    final_df.to_csv(

        tmp_file,

        index=False,

        encoding="utf-8-sig"

    )



    os.replace(tmp_file, OUTPUT_FILE)



    print("=" * 100)

    print("[DONE] 更新完成")

    print("保存路径:", OUTPUT_FILE)

    print("总行数:", len(final_df))

    print("有 market_cap 数量:", final_df["market_cap"].notna().sum())

    print("无 market_cap 数量:", final_df["market_cap"].isna().sum())

    print("市值 > 300M 数量:", (final_df["market_cap"] > 300_000_000).sum())

    print("市值 > 1B 数量:", (final_df["market_cap"] > 1_000_000_000).sum())

    print("=" * 100)





if __name__ == "__main__":

    main()

