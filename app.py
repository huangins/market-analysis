import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面配置
st.set_page_config(page_title="多維度量化趨勢回測儀表板", layout="wide")
st.title("📈 槓桿 ETF 多維度趨勢策略與動態回測儀表板")

# 2. 側邊欄控制項
st.sidebar.header("⚙️ 標的與策略參數設定")

preset_options = [
    "SPY", "QLD", "QQQ", "00631L.TW",
    "NVDA", "NVDL", "TSM", "TSMX",
    "GOOG", "GGLL", "META", "FBL",
    "MU", "MULL", "MRVL", "自訂代號..."
]
selected_preset = st.sidebar.selectbox("選擇預設標的", preset_options, index=1)

if selected_preset == "自訂代號...":
    ticker = st.sidebar.text_input("輸入股票/ETF 代號 (如 AAPL, SPY, 2330.TW)", "AAPL").strip().upper()
else:
    ticker = selected_preset

start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("today"))

st.sidebar.markdown("---")
st.sidebar.subheader("技術指標參數設定")
fast_ma = st.sidebar.slider("快線天數 (Fast MA)", 10, 60, 20)
slow_ma = st.sidebar.slider("慢線天數 (Slow MA)", 20, 300, 200)
adx_period = st.sidebar.slider("ADX 週期", 7, 21, 14)
adx_threshold = st.sidebar.slider("ADX 趨勢門檻 (小於此值無視均線黃金交叉)", 10, 35, 25)

# 3. 策略層級說明卡
with st.expander("📖 策略層級 (Level 1 / Level 2 / Level 3) 運作邏輯說明", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 🔹 Level 1: Buy & Hold")
        st.caption("**無腦持續持有**：100% 資金始終滿倉。吃滿多頭利潤，但也必須 100% 硬扛大熊市的深幅回撤。")
    with col_b:
        st.markdown("#### 🔹 Level 2: 純雙均線")
        st.caption(f"**純價格趨勢（{fast_ma}MA / {slow_ma}MA）**：快線 > 慢線時持倉；快線 < 慢線時避險。缺點是無方向盤整期會被頻繁雙巴。")
    with col_c:
        st.markdown("#### 🔹 Level 3: 多維度過濾系統")
        st.caption(f"**均線 + ADX 風控濾網**：快線 < 慢線時**無條件賣出**；快線 > 慢線且 **ADX ≥ {adx_threshold}** 才允許買入，完美避開盤整期。")

# 4. 動態計算 Warm-up 天數並下載數據
required_warmup_days = int(slow_ma * 2.2) + 60

def fetch_data(symbol, start, end, warmup_days):
    warmup_start = pd.to_datetime(start) - pd.Timedelta(days=warmup_days)
    df_raw = yf.download(symbol, start=warmup_start, end=end)
    return df_raw, pd.to_datetime(start)

df = None
with st.status(f"正在擷取與計算 `{ticker}` 量化指標...", expanded=True) as status:
    st.write(f"📡 從 Yahoo Finance API 載入 `{ticker}` 歷史 K 線 (動態預熱 {required_warmup_days} 天)...")
    df_raw, filter_start_dt = fetch_data(ticker, start_date, end_date, required_warmup_days)

    if df_raw.empty:
        status.update(label=f"❌ 載入失敗：無法取得 `{ticker}` 數據，請確認代號正確。", state="error", expanded=True)
        st.stop()
    else:
        df = df_raw.copy()

        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.levels[1]:
                df = df.xs(ticker, level=1, axis=1)
            else:
                df = df.droplevel(1, axis=1)

        st.write("⚙️ 運算 Fast MA, Slow MA 與 ADX (Wilder's EMA) 指標...")
        df['SMA_Fast'] = df['Close'].rolling(window=fast_ma).mean()
        df['SMA_Slow'] = df['Close'].rolling(window=slow_ma).mean()

        # 標準 Wilder's ADX 算法 (EMA 平滑)
        alpha = 1 / adx_period
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

        up = df['High'] - df['High'].shift(1)
        down = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((up > down) & (up > 0), up, 0)
        df['-DM'] = np.where((down > up) & (down > 0), down, 0)

        df['+DI'] = 100 * (df['+DM'].ewm(alpha=alpha, adjust=False).mean() / df['ATR'])
        df['-DI'] = 100 * (df['-DM'].ewm(alpha=alpha, adjust=False).mean() / df['ATR'])

        di_sum = df['+DI'] + df['-DI']
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / np.where(di_sum == 0, 1, di_sum)
        df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()

        df['Ret'] = df['Close'].pct_change()
        df['Pos_BH'] = 1.0
        df['Strat_BH'] = df['Ret'] * df['Pos_BH']
        df['Pos_L2'] = np.where(df['SMA_Fast'] > df['SMA_Slow'], 1.0, 0.0)
        df['Strat_L2'] = df['Ret'] * df['Pos_L2'].shift(1)

        # 嚴謹的 Level 3 狀態機
        pos_l3 = []
        curr_pos = 0.0
        for i in range(len(df)):
            fast = df['SMA_Fast'].iloc[i]
            slow = df['SMA_Slow'].iloc[i]
            adx_val = df['ADX'].iloc[i]

            if pd.isna(slow) or pd.isna(adx_val):
                curr_pos = 0.0
            elif fast < slow:
                curr_pos = 0.0
            elif fast > slow and adx_val >= adx_threshold:
                curr_pos = 1.0

            pos_l3.append(curr_pos)

        df['Pos_L3'] = pd.Series(pos_l3, index=df.index)
        df['Strat_L3'] = df['Ret'] * df['Pos_L3'].shift(1)

        df['Trade_L2_Sig'] = df['Pos_L2'].diff()
        df['Trade_L3_Sig'] = df['Pos_L3'].diff()

        df = df[df.index >= filter_start_dt].copy()

        first_slow_val = df['SMA_Slow'].iloc[0] if not df.empty else np.nan
        if pd.isna(first_slow_val):
            st.warning(f"⚠️ 注意：`{ticker}` 在選定起點日期的歷史資料長度不足計算 {slow_ma} 日均線。前段無數據區間將暫時顯示為空手狀態。")

        df['Cum_BH'] = (1 + df['Strat_BH'].fillna(0)).cumprod()
        df['Cum_L2'] = (1 + df['Strat_L2'].fillna(0)).cumprod()
        df['Cum_L3'] = (1 + df['Strat_L3'].fillna(0)).cumprod()

        status.update(label=f"✅ `{ticker}` 數據與指標運算完成！", state="complete", expanded=False)

# 5. 數據績效摘要
def get_metrics(cum_series):
    valid = cum_series.dropna()
    if len(valid) < 2:
        return 0.0, 0.0
    total_ret = valid.iloc[-1] - 1
    days = (valid.index[-1] - valid.index[0]).days
    cagr = ((1 + total_ret) ** (365 / max(days, 1))) - 1
    peak = valid.cummax()
    dd = (valid - peak) / peak
    return cagr, dd.min()

cagr_bh, mdd_bh = get_metrics(df['Cum_BH'])
cagr_l2, mdd_l2 = get_metrics(df['Cum_L2'])
cagr_l3, mdd_l3 = get_metrics(df['Cum_L3'])

trades_l2 = int(df['Trade_L2_Sig'].abs().sum() / 2)
trades_l3 = int(df['Trade_L3_Sig'].abs().sum() / 2)

col1, col2, col3 = st.columns(3)
col1.metric("Level 1: Buy & Hold", f"{cagr_bh:.1%} (CAGR)", delta=f"MDD: {mdd_bh:.1%}", delta_color="off")
col2.metric("Level 2: 純雙均線", f"{cagr_l2:.1%} (CAGR)", delta=f"MDD: {mdd_l2:.1%} | 交易 {trades_l2} 次", delta_color="off")
col3.metric("Level 3: 均線 + ADX", f"{cagr_l3:.1%} (CAGR)", delta=f"MDD: {mdd_l3:.1%} | 交易 {trades_l3} 次", delta_color="off")

st.markdown("---")

# 6. 多圖聯動 Plotly 儀表板
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        f"{ticker} 價格圖：Level 2 (◯買/✕賣) vs Level 3 (▲買/▼賣) 訊號對比",
        f"ADX 趨勢強度 (門檻值 = {adx_threshold}；低於此值 Level 3 不發動新買單)",
        "資產累積倍數曲線比較 (Equity Curves)"
    ),
    row_heights=[0.45, 0.25, 0.3]
)

# Subplot 1: Price & MAs
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="收盤價", line=dict(color='#F3F4F6', width=1.8)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Fast'], name=f"快線({fast_ma}MA)", line=dict(color='#00F0FF', width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Slow'], name=f"慢線({slow_ma}MA)", line=dict(color='#FF007F', width=1.5)), row=1, col=1)

# Level 2 買賣訊號點標記
buy_l2 = df[df['Trade_L2_Sig'] == 1.0]
sell_l2 = df[df['Trade_L2_Sig'] == -1.0]

fig.add_trace(go.Scatter(
    x=buy_l2.index, y=buy_l2['Close'], mode='markers', name='Level 2 買入 (純均線)',
    marker=dict(symbol='circle-open', size=9, color='#00F0FF', line=dict(width=2))
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=sell_l2.index, y=sell_l2['Close'], mode='markers', name='Level 2 賣出 (純均線)',
    marker=dict(symbol='x', size=9, color='#FF007F', line=dict(width=2))
), row=1, col=1)

# Level 3 買賣訊號點標記
buy_l3 = df[df['Trade_L3_Sig'] == 1.0]
sell_l3 = df[df['Trade_L3_Sig'] == -1.0]

fig.add_trace(go.Scatter(
    x=buy_l3.index, y=buy_l3['Close'], mode='markers', name='Level 3 買入 (均線+ADX)',
    marker=dict(symbol='triangle-up', size=12, color='#00FF66')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=sell_l3.index, y=sell_l3['Close'], mode='markers', name='Level 3 賣出 (均線+ADX)',
    marker=dict(symbol='triangle-down', size=12, color='#FF3333')
), row=1, col=1)

# Subplot 2: ADX
fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name="ADX 趨勢值", line=dict(color='#FFBF00', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df.index, y=[adx_threshold] * len(df), name=f"ADX 門檻 ({adx_threshold})",
    line=dict(color='#9CA3AF', dash='dash')
), row=2, col=1)

# Subplot 3: Equity Curves
fig.add_trace(go.Scatter(x=df.index, y=df['Cum_BH'], name="Level 1 (Buy & Hold)", line=dict(color='#9CA3AF', dash='dot')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Cum_L2'], name="Level 2 (純雙均線)", line=dict(color='#00F0FF', width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Cum_L3'], name="Level 3 (均線+ADX)", line=dict(color='#00F0FF', width=2.5)), row=3, col=1) # 修正 Level 3 顏色為亮綠色與下方一致
# 修正 Level 3 顏色在圖例與線條上保持一致 (#00FF66)
fig.data[-1].line.color = '#00FF66'

# ✅ 佈局與圖例排版徹底優化
fig.update_layout(
    height=950,
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#E5E7EB"),
    hoverlabel=dict(bgcolor="#1F2937", font_color="#F9FAFB", bordercolor="#4B5563"),
    hovermode="x unified",
    # 將圖例移至圖表下方，採用水平橫向排列並支援自動換行，徹底解決頂部擁擠與看不見的問題
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.08,
        xanchor="center",
        x=0.5,
        font=dict(color="#F9FAFB", size=12),
        bgcolor="rgba(17, 24, 39, 0.8)",
        bordercolor="#374151",
        borderwidth=1
    ),
    margin=dict(b=120) # 預留底部圖例空間
)

# ✅ 座標軸網格線淡化處理
fig.update_xaxes(
    showspikes=True, spikecolor="#9CA3AF", spikethickness=1, spikedash="dot", spikemode="across",
    gridcolor="#374151", zerolinecolor="#4B5563"
)
fig.update_yaxes(
    gridcolor="#374151", zerolinecolor="#4B5563"
)

# 單獨修正 Subplot Title 的顏色
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color="#E5E7EB", size=13)

st.plotly_chart(fig, use_container_width=True)

# 最新狀態提示
latest_l3 = df['Pos_L3'].iloc[-1]
latest_l2 = df['Pos_L2'].iloc[-1]
st.info(f"截至最新交易日 (`{df.index[-1].strftime('%Y-%m-%d')}`) — **Level 2 狀態**：{'🟢 多頭持倉' if latest_l2==1 else '🔴 空頭避險'} ｜ **Level 3 狀態**：{'🟢 多頭持倉' if latest_l3==1 else '🔴 空頭避險'}")