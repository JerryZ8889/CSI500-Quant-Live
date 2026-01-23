import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ==========================================
# 1. 网页配置与视觉注入 (Professional UI)
# ==========================================
st.set_page_config(page_title="中证500量化实战决策中心", layout="wide", initial_sidebar_state="expanded")

# 注入 CSS：打造金融终端质感
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #e11d48; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 字体处理 ---
font_path = './csi500_data/SimHei.ttf'
my_font = None
if os.path.exists(font_path):
    try:
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['axes.unicode_minus'] = False 
    except: pass

# ==========================================
# 2. 侧边栏：动态参数调节
# ==========================================
with st.sidebar:
    st.header("⚙️ 策略参数配置")
    date_range = st.date_input("回测时间跨度", 
                               [pd.to_datetime("2024-01-01"), pd.to_datetime("2026-01-15")])
    st.divider()
    ma_window = st.slider("均线趋势过滤窗口 (MA)", 10, 60, 30)
    heat_window = st.slider("成交量热度窗口", 5, 40, 20)
    st.divider()
    st.info("💡 建议：宽幅震荡市调大均线窗口，快速反弹市调小窗口。")

if len(date_range) == 2:
    BACKTEST_START, BACKTEST_END = date_range[0].strftime('%Y-%m-%d'), date_range[1].strftime('%Y-%m-%d')
else:
    BACKTEST_START, BACKTEST_END = "2024-01-01", "2026-01-15"

# ==========================================
# 3. 数据与引擎 (逻辑保持不动)
# ==========================================
@st.cache_data
def load_data(s_date, e_date, ma_win, h_win):
    path_prefix = "./csi500_data/"
    df_index = pd.read_csv(f"{path_prefix}sh.000905.csv") 
    df_breadth = pd.read_csv(f"{path_prefix}csi500_breadth_daily.csv") 
    df_master = pd.read_csv(f"{path_prefix}CSI500_Master_Strategy.csv")

    for df in [df_index, df_breadth, df_master]:
        df['date'] = pd.to_datetime(df['date'])

    df = pd.merge(df_index, df_breadth[['date', 'breadth']], on='date', how='inner')
    df = pd.merge(df, df_master[['date', 'new_high_pct', 'ETF_Turnover']], on='date', how='left')
    
    # ETF 换手处理
    etf_codes = ["510050", "510300", "510500", "512100"]
    for code in etf_codes:
        f_path = f"{path_prefix}{code}.csv"
        if os.path.exists(f_path):
            etf_df = pd.read_csv(f_path)
            etf_df['date'] = pd.to_datetime(etf_df['date'])
            df = pd.merge(df, etf_df[['date', 'turnover']], on='date', how='left', suffixes=('', f'_{code}'))
            df[f'turnover_{code}'] = df[f'turnover_{code}'].fillna(0)
        else: df[f'turnover_{code}'] = 0
    
    df['new_high_pct'] = df['new_high_pct'].fillna(0)
    df['MA_Filter'] = df['close'].rolling(ma_win).mean() 
    df['MA_Trend'] = df['close'].rolling(10).mean()
    df['MA_Support'] = df['close'].rolling(5).mean()
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Streak'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Streak'], 0)
    df['Heat_Z'] = (df['amount'] - df['amount'].rolling(h_win).mean()) / df['amount'].rolling(h_win).std()
    
    t_raw = df['ETF_Turnover']
    df['Turnover_Pct'] = np.where(t_raw.max() > 1, t_raw, t_raw * 100)
    
    return df.sort_values('date').set_index('date').loc[s_date:e_date]

def run_strategy(df_main):
    temp = df_main.copy()
    temp['pos'], temp['signal'] = 0, 0
    in_pos, logic_state, entry_idx, entry_high = False, "", 0, 0

    cond_comp_b = (temp['breadth'] < 16)
    cond_comp_s = (temp['breadth'] > 79) & (temp['Heat_Z'] < 1.5)
    cond_fn_b_base = (temp['close'] > temp['MA_Trend']) & \
                      (temp['Consec_Gains'].shift(1) >= 3) & \
                      (temp['close'] < temp['close'].shift(1)) & \
                      (temp['Turnover_Pct'] > 1.0) & \
                      (temp['close'] > temp['MA_Support'])

    for i in range(len(temp)):
        if i == 0: continue
        curr_close, prev_close, ma30 = temp['close'].iloc[i], temp['close'].iloc[i-1], temp['MA_Filter'].iloc[i]
        if in_pos:
            if logic_state == "FirstNeg" and cond_comp_b.iloc[i]: logic_state = "Composite"
            exit_f = False
            is_1d, is_below_ma = (curr_close < prev_close), (curr_close < ma30)
            is_5d = (i - entry_idx >= 5) and not (temp['close'].iloc[entry_idx:i+1] > entry_high).any()
            if logic_state == "Composite":
                if cond_comp_s.iloc[i]: exit_f = True
            else:
                if cond_comp_s.iloc[i]: exit_f = True
                elif is_below_ma and (is_1d or is_5d): exit_f = True
            if exit_f:
                temp.iloc[i, temp.columns.get_loc('pos')], temp.iloc[i, temp.columns.get_loc('signal')] = 0, -1
                in_pos, logic_state = False, ""
            else: temp.iloc[i, temp.columns.get_loc('pos')] = 1
        else:
            buy_trig = False
            if cond_comp_b.iloc[i]: logic_state, buy_trig = "Composite", True
            elif cond_fn_b_base.iloc[i] and (curr_close > ma30): logic_state, buy_trig = "FirstNeg", True
            if buy_trig:
                temp.iloc[i, temp.columns.get_loc('pos')], temp.iloc[i, temp.columns.get_loc('signal')] = 1, 1
                in_pos, entry_idx, entry_high = True, i, temp['high'].iloc[i]

    temp['strat_ret'] = temp['pos'].shift(1).fillna(0) * temp['close'].pct_change().fillna(0) - np.where(temp['pos'].shift(1).diff() != 0, 0.001, 0)
    temp['cum_ret'] = (1 + temp['strat_ret']).cumprod()
    return temp

# 数据加载
df_input = load_data(BACKTEST_START, BACKTEST_END, ma_window, heat_window)
res = run_strategy(df_input)
res_bench = (1 + df_input['close'].pct_change().fillna(0)).cumprod()

# ==========================================
# 4. 终端级展示 (Professional Dashboard)
# ==========================================

st.title("🛡️ 中证500量化实战决策中心")
st.caption(f"回测周期: {BACKTEST_START} 至 {BACKTEST_END} | 均线过滤: {ma_window}日 | 当前逻辑: 宏观动能+广度错位")

# --- A. 核心绩效看板 (Grid) ---
def get_stats(cum_series):
    total = (cum_series.iloc[-1] - 1) * 100
    mdd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min() * 100
    days = (cum_series.index[-1] - cum_series.index[0]).days
    ann = ((cum_series.iloc[-1])**(365.25/days) - 1) * 100 if days > 0 else 0
    return total, ann, mdd

s_tot, s_ann, s_mdd = get_stats(res['cum_ret'])
b_tot, b_ann, b_mdd = get_stats(res_bench)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚀 策略累计收益", f"{s_tot:.2f}%", f"年化 {s_ann:.2f}%")
c2.metric("📉 策略最大回撤", f"{s_mdd:.2f}%")
c3.metric("🏛️ 基准累计收益", f"{b_tot:.2f}%", f"年化 {b_ann:.2f}%", delta_color="inverse")
c4.metric("📊 相对超额收益", f"{s_tot - b_tot:.2f}%", "Alpha")

st.divider()

# --- B. 数据联动透视 (Tabs Layout) ---
st.subheader("📊 策略全维度分析")
tab1, tab2, tab3 = st.tabs(["🔥 绩效表现 & 信号", "🌐 市场广度 & 热度", "💧 流动性 & 风格对比"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(res_bench, label='中证500基准', color='#94a3b8', alpha=0.4, linestyle='--')
    ax1.plot(res['cum_ret'], label='实战版策略', color='#e11d48', linewidth=2)
    for sig, col, mark in [(1, '#ef4444', '^'), (-1, '#22c55e', 'v')]:
        pts = res[res['signal'] == sig]
        ax1.scatter(pts.index, res.loc[pts.index, 'cum_ret'], color=col, marker=mark, s=150, zorder=5)
    if my_font: ax1.set_title("策略绩效曲线与买卖信号点分布", fontproperties=my_font, fontsize=16)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    # 广度
    ax2.plot(res.index, res['breadth'], color='#f59e0b', label='MA20上方占比 (%)')
    ax2.fill_between(res.index, 0, 100, where=(res['pos']==1), color='#3b82f6', alpha=0.1)
    if my_font: ax2.set_title("市场广度 (Breadth) 演变", fontproperties=my_font)
    # 热度
    ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']>=0), color='#ef4444', alpha=0.5)
    ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']<0), color='#3b82f6', alpha=0.5)
    ax3.axhline(y=1.5, color='#d97706', linestyle='--', label='过热线')
    if my_font: ax3.set_title("资金成交热度 (Heat Z-Score)", fontproperties=my_font)
    st.pyplot(fig2)

with tab3:
    fig3, (ax4, ax5) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    # 新高
    ax4_r = ax4.twinx()
    ax4.plot(res.index, res['breadth'], color='#0f172a', alpha=0.3)
    ax4_r.bar(res.index, res['new_high_pct'], color='#fbbf24', alpha=0.6, label='60日新高占比')
    if my_font: ax4.set_title("季度强度 (60日新高比例)", fontproperties=my_font)
    # ETF对比
    colors = ['#1e40af', '#166534', '#991b1b', '#6b21a8']
    labels = ["上证50", "沪深300", "中证500", "中证1000"]
    etf_cols = [f'turnover_{c}' for c in ["510050", "510300", "510500", "512100"]]
    for i, col in enumerate(etf_cols):
        ax5.plot(res.index, res[col], label=labels[i], color=colors[i], alpha=0.8)
    if my_font: ax5.set_title("核心风格 ETF 换手率监控", fontproperties=my_font)
    ax5.legend(ncol=4)
    st.pyplot(fig3)

st.divider()

# --- C. 实战决策总结 (Command Panel) ---
st.subheader("📋 战术指令板")
latest = res.iloc[-1]
prev = res.iloc[-2]

# 模式判定逻辑
if latest['close'] > latest['MA_Filter'] and latest['MA_Filter'] > prev['MA_Filter']:
    mode, mode_color = "🐂 多头强趋势", "green"
elif latest['close'] < latest['MA_Filter'] and latest['MA_Filter'] < prev['MA_Filter']:
    mode, mode_color = "🐻 空头弱趋势", "red"
else:
    mode, mode_color = "🦓 震荡整理期", "orange"

# 指令判定
signal, pos = latest['signal'], latest['pos']
if signal == 1: action, status = "🚨 执行买入", "success"
elif signal == -1: action, status = "🚨 执行卖出", "error"
elif pos == 1: action, status = "💎 持股待涨", "info"
else: action, status = "🛡️ 空仓等待", "secondary"

c_left, c_right = st.columns([1, 2])
with c_left:
    st.write(f"**当前市场模式：** :{mode_color}[{mode}]")
    if status == "success": st.success(f"### 指令：{action}")
    elif status == "error": st.error(f"### 指令：{action}")
    elif status == "info": st.info(f"### 指令：{action}")
    else: st.warning(f"### 指令：{action}")

with c_right:
    # 逻辑扫描
    logic_desc = []
    if latest['breadth'] < 16: logic_desc.append("📉 [反转信号] 广度已进入冰点区 ( <16% )")
    if latest['Heat_Z'] > 1.5: logic_desc.append("🔥 [预警信号] 资金热度异常过热 ( Z > 1.5 )")
    if latest['new_high_pct'] > 5: logic_desc.append("💪 [强度信号] 季度走强个股显著增多")
    
    st.markdown("**实时风控/机会扫描：**")
    if logic_desc:
        for item in logic_desc: st.write(item)
    else: st.write("✅ 暂无极端信号，维持系统现状。")
    st.write(f"指标快照：广度 {latest['breadth']:.1f}% | 热度 {latest['Heat_Z']:.2f} | 新高比例 {latest['new_high_pct']:.2f}%")
