import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil

# ==========================================
# 1. 网页基础配置与视觉美化 (大师级 UI)
# ==========================================
st.set_page_config(
    page_title="中证500量化实战决策中心", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 注入自定义 CSS 提升质感 ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    /* 指标卡片美化 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    /* 标题加粗 */
    .main-title { font-weight: 800; color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# --- 字体彻底修复逻辑 (暴力清缓存 + 强制注入) ---
cache_dir = os.path.expanduser('~/.cache/matplotlib')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)

font_path = './csi500_data/SimHei.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    my_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = my_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    my_font = None
    st.error("⚠️ 未找到 SimHei.ttf，请检查路径！")

# ==========================================
# 2. 核心参数与数据引擎 (保持原逻辑)
# ==========================================
BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2026-01-15"
MA_FILTER_WINDOW = 30
HEAT_WINDOW = 20

@st.cache_data
def load_data():
    path_prefix = "./csi500_data/"
    df_index = pd.read_csv(f"{path_prefix}sh.000905.csv") 
    df_breadth = pd.read_csv(f"{path_prefix}csi500_breadth_daily.csv") 
    df_master = pd.read_csv(f"{path_prefix}CSI500_Master_Strategy.csv")

    for df in [df_index, df_breadth, df_master]:
        df['date'] = pd.to_datetime(df['date'])

    df = pd.merge(df_index, df_breadth[['date', 'breadth']], on='date', how='inner')
    df = pd.merge(df, df_master[['date', 'new_high_pct', 'ETF_Turnover']], on='date', how='left')
    
    etf_codes = ["510050", "510300", "510500", "512100"]
    for code in etf_codes:
        f_path = f"{path_prefix}{code}.csv"
        if os.path.exists(f_path):
            etf_df = pd.read_csv(f_path)
            etf_df['date'] = pd.to_datetime(etf_df['date'])
            etf_df = etf_df.rename(columns={'turnover': f'turn_raw_{code}'})
            df = pd.merge(df, etf_df[['date', f'turn_raw_{code}']], on='date', how='left')
            df[f'turnover_{code}'] = np.where(df[f'turn_raw_{code}'].max() > 1, 
                                             df[f'turn_raw_{code}'], 
                                             df[f'turn_raw_{code}'] * 100)
        else:
            df[f'turnover_{code}'] = 0
    
    df['new_high_pct'] = df['new_high_pct'].fillna(0)
    df['MA_Filter'] = df['close'].rolling(MA_FILTER_WINDOW).mean() 
    df['MA_Trend'] = df['close'].rolling(10).mean()
    df['MA_Support'] = df['close'].rolling(5).mean()
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Streak'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Streak'], 0)
    df['Heat_Z'] = (df['amount'] - df['amount'].rolling(HEAT_WINDOW).mean()) / df['amount'].rolling(HEAT_WINDOW).std()
    
    t_raw = df['ETF_Turnover']
    df['Turnover_Pct'] = np.where(t_raw.max() > 1, t_raw, t_raw * 100)
    
    return df.sort_values('date').set_index('date').loc[BACKTEST_START:BACKTEST_END]

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
            is_1d = (curr_close < prev_close)
            is_below_ma = (curr_close < ma30)
            is_5d = (i - entry_idx >= 5) and not (temp['close'].iloc[entry_idx:i+1] > entry_high).any()
            
            if logic_state == "Composite":
                if cond_comp_s.iloc[i]: exit_f = True
            else:
                if cond_comp_s.iloc[i]: exit_f = True
                elif is_below_ma and (is_1d or is_5d): exit_f = True
            
            if exit_f:
                temp.iloc[i, temp.columns.get_loc('pos')], temp.iloc[i, temp.columns.get_loc('signal')] = 0, -1
                in_pos, logic_state = False, ""
            else:
                temp.iloc[i, temp.columns.get_loc('pos')] = 1
        else:
            buy_trig = False
            if cond_comp_b.iloc[i]: logic_state, buy_trig = "Composite", True
            elif cond_fn_b_base.iloc[i] and (curr_close > ma30): logic_state, buy_trig = "FirstNeg", True
            if buy_trig:
                temp.iloc[i, temp.columns.get_loc('pos')], temp.iloc[i, temp.columns.get_loc('signal')] = 1, 1
                in_pos, entry_idx, entry_high = True, i, temp['high'].iloc[i]

    actual_pos = temp['pos'].shift(1).fillna(0)
    temp['strat_ret'] = actual_pos * temp['close'].pct_change().fillna(0) - np.where(actual_pos.diff() != 0, 0.001, 0)
    temp['cum_ret'] = (1 + temp['strat_ret']).cumprod()
    return temp

# 数据加载与计算
df_input = load_data()
res = run_strategy(df_input)
res_bench = (1 + df_input['close'].pct_change().fillna(0)).cumprod()

# ==========================================
# 3. UI 展示区 (精美重构版)
# ==========================================
st.markdown("<h1 class='main-title'>🛡️ 中证500量化实战决策中心</h1>", unsafe_allow_html=True)

# 3.1 绩效统计卡片
st.subheader("📊 策略绩效统计")
def get_stats(cum_series):
    total = (cum_series.iloc[-1] - 1) * 100
    mdd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min() * 100
    days = (cum_series.index[-1] - cum_series.index[0]).days
    ann = ((cum_series.iloc[-1])**(365.25/days) - 1) * 100 if days > 0 else 0
    return total, ann, mdd

s_tot, s_ann, s_mdd = get_stats(res['cum_ret'])
b_tot, b_ann, b_mdd = get_stats(res_bench)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🚀 策略收益", f"{s_tot:.2f}%", f"年化 {s_ann:.2f}%")
with c2: st.metric("📉 策略回撤", f"{s_mdd:.2f}%")
with c3: st.metric("🏛️ 基准收益", f"{b_tot:.2f}%", f"年化 {b_ann:.2f}%", delta_color="inverse")
with c4: st.metric("🌊 基准回撤", f"{b_mdd:.2f}%")

st.divider()

# 3.2 图表可视化 (注入本地字体防止乱码)
st.subheader("📈 全维度数据视图")
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 32), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 0.8, 0.8, 1.2, 1.2]})

# 图1: 收益与信号
ax1.plot(res_bench, label='中证500基准', color='#94a3b8', alpha=0.4, linestyle='--')
ax1.plot(res['cum_ret'], label='实战版策略', color='#e11d48', linewidth=2.5)
for sig, col, mark in [(1, '#ef4444', '^'), (-1, '#22c55e', 'v')]:
    pts = res[res['signal'] == sig]
    ax1.scatter(pts.index, res.loc[pts.index, 'cum_ret'], color=col, marker=mark, s=180, zorder=5, edgecolors='white')
ax1.set_title("策略绩效与实战信号分布", fontproperties=my_font, fontsize=16); ax1.legend(prop=my_font)

# 图2: 广度
ax2.plot(res.index, res['breadth'], color='#f59e0b', label='MA20上方占比 (%)')
ax2.fill_between(res.index, 0, 100, where=(res['pos']==1), color='#3b82f6', alpha=0.1)
ax2.set_title("市场广度波动环境", fontproperties=my_font); ax2.set_ylim(0, 100)

# 图3: 热度
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']>=0), color='#ef4444', alpha=0.5)
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']<0), color='#3b82f6', alpha=0.5)
ax3.axhline(y=1.5, color='#d97706', linestyle='--', label='过热线')
ax3.set_title("资金热度 (20日 Z-Score)", fontproperties=my_font); ax3.legend(prop=my_font)

# 图4: 趋势对比
ax4_r = ax4.twinx()
ax4.plot(res.index, res['breadth'], color='#0f172a', linewidth=1.5, label='站上MA20占比')
ax4_r.bar(res.index, res['new_high_pct'], color='#fbbf24', alpha=0.6, width=0.8, label='60日新高占比')
ax4.set_title("市场广度与季度强度趋势对比", fontproperties=my_font)

# 图5: ETF 换手
colors = ['#1e40af', '#166534', '#991b1b', '#6b21a8']
etfs = {"510050": "上证50", "510300": "沪深300", "510500": "中证500", "512100": "中证1000"}
for i, (code, label) in enumerate(etfs.items()):
    ax5.plot(res.index, res[f'turnover_{code}'], label=f"{label} 换手率", color=colors[i], alpha=0.8)
ax5.set_title("核心风格 ETF 换手率对比", fontproperties=my_font); ax5.legend(prop=my_font, ncol=4)

plt.tight_layout()
st.pyplot(fig)

st.divider()

# 3.3 实战决策总结 (模块找回并深度美化)
st.subheader("📝 实战决策总结")

latest = res.iloc[-1]
prev = res.iloc[-2]

# A. 市场模式判定
if latest['close'] > latest['MA_Filter'] and latest['MA_Filter'] > prev['MA_Filter']:
    mode, mode_desc = "🐂 多头强趋势", "价格站上MA30且均线向上，适合积极博弈"
elif latest['close'] < latest['MA_Filter'] and latest['MA_Filter'] < prev['MA_Filter']:
    mode, mode_desc = "🐻 空头弱趋势", "价格处于MA30下方且均线向下，风险较高"
else:
    mode, mode_desc = "🦓 震荡过渡期", "价格与均线纠缠，建议耐心观察信号"

# B. 操作建议判定
signal, pos = latest['signal'], latest['pos']
if signal == 1: 
    action, status_type = "🚨 立即买入信号", "success"
elif signal == -1: 
    action, status_type = "🚨 立即卖出信号", "error"
elif pos == 1: 
    action, status_type = "💎 持股待涨", "info"
else: 
    action, status_type = "🛡️ 空仓观望", "secondary"

# C. 逻辑详情
logic_desc = []
if latest['breadth'] < 16: logic_desc.append("📉 市场广度已跌破16%，进入博弈冰点区")
if latest['Heat_Z'] > 1.5: logic_desc.append("🔥 资金热度异常，需警惕短期过热风险")
if latest['new_high_pct'] > 5: logic_desc.append("💪 创60日新高个股占比显著，内生动力增强")

# 展示布局
col_sum1, col_sum2 = st.columns([1, 1.5])

with col_sum1:
    st.markdown(f"**1. 当前模式：** {mode}")
    st.caption(mode_desc)
    st.markdown(f"**2. 操作建议：**")
    if status_type == "success": st.success(action)
    elif status_type == "error": st.error(action)
    elif status_type == "info": st.info(action)
    else: st.warning(action)

with col_sum2:
    st.markdown("**3. 数据核心指标：**")
    st.write(f"资金热度：{latest['Heat_Z']:.2f} (Z-Score)")
    st.write(f"市场广度：{latest['breadth']:.2f}% | 60日新高比例：{latest['new_high_pct']:.2f}%")
    st.markdown("**4. 逻辑扫描：**")
    for item in (logic_desc if logic_desc else ["✅ 目前处于常规波动区间，无异常逻辑触发"]):
        st.write(f"- {item}")
