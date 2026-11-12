import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面配置
st.set_page_config(page_title="多維度量化趨勢回測儀表板", layout="wide")
st.title("📈 槓桿 ETF 跨標的訊號 & 綜合動能回測系統")

# 2. 側邊欄控制項
st.sidebar.header("⚙️ 標的設定 (跨標的訊號)")

# 還原 1 倍與槓桿標的
preset_trade = [
    "SPY", "QQQ", "QLD", "TQQQ", "00631L.TW",
    "NVDA", "NVDL", "TSM", "TSMX",
    "GOOG", "GGLL", "META", "FBL",
    "MU", "MULL", "MRVL", "自訂代號..."
]
selected_trade = st.sidebar.selectbox("💰 1. 選擇「交易標的」(買賣部位)", preset_trade, index=6) # 預設 NVDL
if selected_trade == "自訂代號...":
    trade_ticker = st.sidebar.text_input("輸入交易標的代號", "NVDL").strip().upper()
else:
    trade_ticker = selected_trade

preset_signal = ["同交易標的 (預設)", "NVDA", "SOXX", "QQQ", "TSM", "SPY", "自訂代號..."]
selected_signal = st.sidebar.selectbox("📡 2. 選擇「訊號標的」(判定均線/ATR/MFI)", preset_signal, index=1)
if selected_signal == "同交易標的 (預設)":
    signal_ticker = trade_ticker
elif selected_signal == "自訂代號...":
    signal_ticker = st.sidebar.text_input("輸入訊號標的代號", "NVDA").strip().upper()
else:
    signal_ticker = selected_signal

start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("today"))

st.sidebar.markdown("---")
st.sidebar.subheader("技術指標參數設定")
fast_ma = st.sidebar.slider("快線天數 (Fast MA)", 10, 60, 20)
slow_ma = st.sidebar.slider("慢線天數 (Slow MA)", 20, 300, 50)
st.sidebar.markdown("---")
atr_period = st.sidebar.slider("ATR 週期 (防守濾網)", 7, 21, 14)
atr_multiplier = st.sidebar.slider("ATR 突破倍數", 1.0, 4.0, 1.5, step=0.1)
st.sidebar.markdown("---")
mfi_period = st.sidebar.slider("MFI 週期 (攻擊濾網)", 7, 21, 14)
mfi_threshold = st.sidebar.slider("MFI 爆量門檻", 20, 80, 50)

# 3. 策略層級說明卡
with st.expander("📖 策略層級 (L2 / L3 / L4 / L3+4) 運作邏輯說明", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🔹 Level 2: 純雙均線")
        st.caption("僅看快慢線交叉。快線 > 慢線做多，反之平倉。")
        st.markdown("#### 🔹 Level 3: 均線 + ATR (防守特化)")
        st.caption("除了均線多頭，**收盤價必須突破 ATR 壓力線上軌**才進場，過濾低波盤整假突破。")
    with col_b:
        st.markdown("#### 🔹 Level 4: 均線 + MFI (攻擊特化)")
        st.caption(f"除了均線多頭，**MFI 必須 ≥ {mfi_threshold} (帶量資金流入)** 才進場，專抓強勢 V 轉。")
        st.markdown("#### 🔹 Level 3+4: 均線 + ATR + MFI (完全體)")
        st.caption("最嚴格的進場條件：必須同時滿足 **突破 ATR 上軌** 且 **MFI 帶量流入**。所有策略的共通出場條件皆為：**快線跌破慢線**即嚴格停損。")

# 4. 動態計算與指標運算
required_warmup_days = int(slow_ma * 2.2) + 60

def fetch_data(symbol, start, end, warmup_days):
    warmup_start = pd.to_datetime(start) - pd.Timedelta(days=warmup_days)
    df_raw = yf.download(symbol, start=warmup_start, end=end)
    if not df_raw.empty:
        if isinstance(df_raw.columns, pd.MultiIndex):
            if symbol in df_raw.columns.levels[1]:
                df_raw = df_raw.xs(symbol, level=1, axis=1)
            else:
                df_raw = df_raw.droplevel(1, axis=1)
    return df_raw, pd.to_datetime(start)

df_trade, df_signal = None, None
with st.status(f"正在擷取並計算量化指標...", expanded=True) as status:
    df_trade, filter_start_dt = fetch_data(trade_ticker, start_date, end_date, required_warmup_days)
    df_signal, _ = fetch_data(signal_ticker, start_date, end_date, required_warmup_days)

    if df_trade.empty or df_signal.empty:
        status.update(label="❌ 載入失敗：無法取得數據。", state="error", expanded=True)
        st.stop()
    else:
        st.write(f"⚙️ 運算 `{signal_ticker}` 的 MA、ATR 與 MFI...")

        # 1. 均線
        df_signal['SMA_Fast'] = df_signal['Close'].rolling(window=fast_ma).mean()
        df_signal['SMA_Slow'] = df_signal['Close'].rolling(window=slow_ma).mean()

        # 2. MFI
        tp = (df_signal['High'] + df_signal['Low'] + df_signal['Close']) / 3
        rmf = tp * df_signal['Volume']
        pmf = np.where(tp > tp.shift(1), rmf, 0.0)
        nmf = np.where(tp < tp.shift(1), rmf, 0.0)
        pmf_sum = pd.Series(pmf, index=df_signal.index).rolling(window=mfi_period).sum()
        nmf_sum = pd.Series(nmf, index=df_signal.index).rolling(window=mfi_period).sum()
        mfr = pmf_sum / np.where(nmf_sum == 0, 1e-9, nmf_sum)
        df_signal['MFI'] = 100 - (100 / (1 + mfr))

        # 3. ATR 與 Supertrend 壓力線
        df_signal['H-L'] = df_signal['High'] - df_signal['Low']
        df_signal['H-PC'] = abs(df_signal['High'] - df_signal['Close'].shift(1))
        df_signal['L-PC'] = abs(df_signal['Low'] - df_signal['Close'].shift(1))
        df_signal['TR'] = df_signal[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df_signal['ATR'] = df_signal['TR'].rolling(window=atr_period).mean()
        # 簡化的 ATR 上軌 (動態壓力線)
        df_signal['ATR_Upper'] = df_signal['SMA_Fast'] + (df_signal['ATR'] * atr_multiplier)

        # 狀態機運算
        df_signal['Pos_L2'] = np.where(df_signal['SMA_Fast'] > df_signal['SMA_Slow'], 1.0, 0.0)

        pos_l3, pos_l4, pos_l34 = [], [], []
        curr_l3, curr_l4, curr_l34 = 0.0, 0.0, 0.0

        for i in range(len(df_signal)):
            close_px = df_signal['Close'].iloc[i]
            fast = df_signal['SMA_Fast'].iloc[i]
            slow = df_signal['SMA_Slow'].iloc[i]
            mfi_val = df_signal['MFI'].iloc[i]
            atr_upper = df_signal['ATR_Upper'].iloc[i]

            if pd.isna(slow) or pd.isna(mfi_val) or pd.isna(atr_upper):
                curr_l3, curr_l4, curr_l34 = 0.0, 0.0, 0.0
            else:
                # L3: 均線 + ATR突破
                if fast < slow: curr_l3 = 0.0
                elif close_px > atr_upper: curr_l3 = 1.0

                # L4: 均線 + MFI帶量
                if fast < slow: curr_l4 = 0.0
                elif (close_px > fast) and (mfi_val >= mfi_threshold): curr_l4 = 1.0

                # L3+4: 均線 + ATR突破 + MFI帶量 (最強防禦與攻擊)
                if fast < slow: curr_l34 = 0.0
                elif (close_px > atr_upper) and (mfi_val >= mfi_threshold): curr_l34 = 1.0

            pos_l3.append(curr_l3)
            pos_l4.append(curr_l4)
            pos_l34.append(curr_l34)

        df_signal['Pos_L3'] = pd.Series(pos_l3, index=df_signal.index)
        df_signal['Pos_L4'] = pd.Series(pos_l4, index=df_signal.index)
        df_signal['Pos_L34'] = pd.Series(pos_l34, index=df_signal.index)

        # 💡 修復映射 Bug：將訊號 DataFrame 與交易 DataFrame 進行 Index 對齊 (Merge)
        # 這樣就不會發生 Reindex 出現全 NaN 導致訊號失效的問題
        df_merged = df_trade[['Close']].join(df_signal[['Pos_L2', 'Pos_L3', 'Pos_L4', 'Pos_L34']], how='left')
        df_merged.ffill(inplace=True) # 填補休市日差值
        df_merged.fillna(0, inplace=True)

        df_merged['Ret'] = df_merged['Close'].pct_change()
        df_merged['Strat_BH'] = df_merged['Ret'] * 1.0
        df_merged['Strat_L2'] = df_merged['Ret'] * df_merged['Pos_L2'].shift(1)
        df_merged['Strat_L3'] = df_merged['Ret'] * df_merged['Pos_L3'].shift(1)
        df_merged['Strat_L4'] = df_merged['Ret'] * df_merged['Pos_L4'].shift(1)
        df_merged['Strat_L34'] = df_merged['Ret'] * df_merged['Pos_L34'].shift(1)

        # 裁切回使用者起點
        df_signal = df_signal[df_signal.index >= filter_start_dt].copy()
        df_merged = df_merged[df_merged.index >= filter_start_dt].copy()

        df_merged['Cum_BH'] = (1 + df_merged['Strat_BH'].fillna(0)).cumprod()
        df_merged['Cum_L2'] = (1 + df_merged['Strat_L2'].fillna(0)).cumprod()
        df_merged['Cum_L3'] = (1 + df_merged['Strat_L3'].fillna(0)).cumprod()
        df_merged['Cum_L4'] = (1 + df_merged['Strat_L4'].fillna(0)).cumprod()
        df_merged['Cum_L34'] = (1 + df_merged['Strat_L34'].fillna(0)).cumprod()

        status.update(label=f"✅ 回測運算完成！", state="complete", expanded=False)

# 5. 數據績效摘要
def get_metrics(cum_series):
    valid = cum_series.dropna()
    if len(valid) < 2: return 0.0, 0.0
    total_ret = valid.iloc[-1] - 1
    days = (valid.index[-1] - valid.index[0]).days
    cagr = ((1 + total_ret) ** (365 / max(days, 1))) - 1
    peak = valid.cummax()
    dd = (valid - peak) / peak
    return cagr, dd.min()

cagr_bh, mdd_bh = get_metrics(df_merged['Cum_BH'])
cagr_l2, mdd_l2 = get_metrics(df_merged['Cum_L2'])
cagr_l3, mdd_l3 = get_metrics(df_merged['Cum_L3'])
cagr_l4, mdd_l4 = get_metrics(df_merged['Cum_L4'])
cagr_l34, mdd_l34 = get_metrics(df_merged['Cum_L34'])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("L1: Buy & Hold", f"{cagr_bh:.1%}", delta=f"MDD: {mdd_bh:.1%}", delta_color="off")
col2.metric("L2: 純雙均線", f"{cagr_l2:.1%}", delta=f"MDD: {mdd_l2:.1%}", delta_color="off")
col3.metric("L3: 均線+ATR", f"{cagr_l3:.1%}", delta=f"MDD: {mdd_l3:.1%}", delta_color="off")
col4.metric("L4: 均線+MFI", f"{cagr_l4:.1%}", delta=f"MDD: {mdd_l4:.1%}", delta_color="off")
col5.metric("L3+4: 均線+ATR+MFI", f"{cagr_l34:.1%}", delta=f"MDD: {mdd_l34:.1%}", delta_color="off")

st.markdown("---")

# 6. 多圖聯動 Plotly 儀表板
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        f"📡 訊號標的 `{signal_ticker}`：均線與 ATR 上軌 (防守壓力線)",
        f"🌊 訊號標的 `{signal_ticker}`：MFI 資金流量 (攻擊濾網)",
        f"💰 交易標的 `{trade_ticker}`：策略淨值曲線比較"
    ),
    row_heights=[0.45, 0.25, 0.3]
)

# Subplot 1: Price, MAs & ATR
fig.add_trace(go.Scatter(x=df_signal.index, y=df_signal['Close'], name=f"{signal_ticker} 收盤價", line=dict(color='#F3F4F6', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_signal.index, y=df_signal['SMA_Fast'], name=f"快線({fast_ma}MA)", line=dict(color='#00F0FF', width=1.0)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_signal.index, y=df_signal['SMA_Slow'], name=f"慢線({slow_ma}MA)", line=dict(color='#FF007F', width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_signal.index, y=df_signal['ATR_Upper'], name=f"ATR 上軌壓力", line=dict(color='#F59E0B', width=1.5, dash='dot')), row=1, col=1)

# 由於訊號種類太多，圖上只畫 L3+4 的終極進出點避免畫面太亂
df_merged['Trade_L34_Sig'] = df_merged['Pos_L34'].diff()
buy_l34 = df_merged[df_merged['Trade_L34_Sig'] == 1.0]
sell_l34 = df_merged[df_merged['Trade_L34_Sig'] == -1.0]

# 將買賣點畫在 Subplot 1 上，Y軸對應訊號標的收盤價
matched_buy = df_signal.loc[df_signal.index.intersection(buy_l34.index)]
matched_sell = df_signal.loc[df_signal.index.intersection(sell_l34.index)]

fig.add_trace(go.Scatter(x=matched_buy.index, y=matched_buy['Close'], mode='markers', name='L3+4 買入', marker=dict(symbol='triangle-up', size=14, color='#00FF66')), row=1, col=1)
fig.add_trace(go.Scatter(x=matched_sell.index, y=matched_sell['Close'], mode='markers', name='L3+4 賣出', marker=dict(symbol='triangle-down', size=14, color='#FF3333')), row=1, col=1)

# Subplot 2: MFI
fig.add_trace(go.Scatter(x=df_signal.index, y=df_signal['MFI'], name="MFI", line=dict(color='#8B5CF6', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df_signal.index, y=[mfi_threshold]*len(df_signal), name="MFI 門檻", line=dict(color='#9CA3AF', dash='dash')), row=2, col=1)

# Subplot 3: Equity Curves
fig.add_trace(go.Scatter(x=df_merged.index, y=df_merged['Cum_BH'], name="L1: B&H", line=dict(color='#6B7280', dash='dot')), row=3, col=1)
fig.add_trace(go.Scatter(x=df_merged.index, y=df_merged['Cum_L2'], name="L2: 純均線", line=dict(color='#00F0FF', width=1.2)), row=3, col=1)
fig.add_trace(go.Scatter(x=df_merged.index, y=df_merged['Cum_L3'], name="L3: +ATR", line=dict(color='#F59E0B', width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df_merged.index, y=df_merged['Cum_L4'], name="L4: +MFI", line=dict(color='#8B5CF6', width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df_merged.index, y=df_merged['Cum_L34'], name="L3+4: 完全體", line=dict(color='#00FF66', width=2.5)), row=3, col=1)

# Layout
fig.update_layout(
    height=950, template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(color="#E5E7EB"), hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5, font=dict(size=12), bgcolor="rgba(17,24,39,0.8)", bordercolor="#374151", borderwidth=1),
    margin=dict(b=120)
)
fig.update_xaxes(showspikes=True, spikecolor="#9CA3AF", spikethickness=1, spikedash="dot", spikemode="across", gridcolor="#374151", zerolinecolor="#4B5563")
fig.update_yaxes(gridcolor="#374151", zerolinecolor="#4B5563")
for annotation in fig['layout']['annotations']: annotation['font'] = dict(color="#E5E7EB", size=14)

st.plotly_chart(fig, use_container_width=True)

# 狀態提示
latest_l34 = df_merged['Pos_L34'].iloc[-1]
st.info(f"截至最新交易日 (`{df_merged.index[-1].strftime('%Y-%m-%d')}`) — **{trade_ticker} 終極 (L3+4) 部位**：{'🟢 多頭持倉 (ATR突破且MFI帶量)' if latest_l34==1 else '🔴 空頭避險 (防守中)'}")