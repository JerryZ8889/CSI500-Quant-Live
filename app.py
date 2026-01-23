import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ==========================================
# 1. 网页基础配置与视觉美化
# ==========================================
st.set_page_config(page_title="中证500量化实战决策中心", layout="wide")

# --- 字体加载（局部注入模式：防崩溃 + 治乱码） ---
font_path = './csi500_data/SimHei.ttf'
my_font = None

if os.path.exists(font_path):
    try:
        # 使用 FontProperties 局部调用，不强制注册全局，防止 RuntimeError
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception as e:
        st.sidebar.warning(f"字体加载异常，将使用系统备用字体: {e}")
else:
    st.sidebar.error("⚠️ 未找到 SimHei.ttf，请检查路径！")

# ==========================================
# 2. 核心参数与数据引擎 (完全保留你的原逻辑)
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

# 数据加载
df_input = load_data()
res = run_strategy(df_input)
res_bench = (1 + df_input['close'].pct_change().fillna(0)).cumprod()

# ==========================================
# 3. UI 展示与可视化 (大师美化版)
# ==========================================
st.title("🛡️ 中证500量化实战决策中心")

# --- A. 策略绩效统计卡片 ---
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
c1.metric("🚀 策略累计收益", f"{s_tot:.2f}%", f"年化 {s_ann:.2f}%")
c2.metric("📉 策略最大回撤", f"{s_mdd:.2f}%")
c3.metric("🏛️ 基准累计收益", f"{b_tot:.2f}%", f"年化 {b_ann:.2f}%", delta_color="inverse")
c4.metric("🌊 基准最大回撤", f"{b_mdd:.2f}%")

st.divider()

# --- B. 全维度数据图表 (注入字体属性解决乱码) ---
st.subheader("📈 全维度数据视图")
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 32), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 0.8, 0.8, 1.2, 1.2]})

def set_font(ax, title):
    if my_font:
        ax.set_title(title, fontproperties=my_font, fontsize=16)
    else:
        ax.set_title(title, fontsize=16)

# 图1: 收益
ax1.plot(res_bench, label='基准', color='#94a3b8', alpha=0.4, linestyle='--')
ax1.plot(res['cum_ret'], label='策略', color='#e11d48', linewidth=2.5)
for sig, col, mark in [(1, '#ef4444', '^'), (-1, '#22c55e', 'v')]:
    pts = res[res['signal'] == sig]
    ax1.scatter(pts.index, res.loc[pts.index, 'cum_ret'], color=col, marker=mark, s=180, zorder=5)
set_font(ax1, "策略绩效与实战信号分布")
if my_font: ax1.legend(prop=my_font)

# 图2: 广度
ax2.plot(res.index, res['breadth'], color='#f59e0b', label='MA20上方占比 (%)')
ax2.fill_between(res.index, 0, 100, where=(res['pos']==1), color='#3b82f6', alpha=0.1)
set_font(ax2, "市场广度波动环境")

# 图3: 热度
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']>=0), color='#ef4444', alpha=0.5)
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']<0), color='#3b82f6', alpha=0.5)
ax3.axhline(y=1.5, color='#d97706', linestyle='--', label='过热线')
set_font(ax3, "资金热度 (20日 Z-Score)")

# 图4: 对比
ax4_r = ax4.twinx()
ax4.plot(res.index, res['breadth'], color='#0f172a', label='广度')
ax4_r.bar(res.index, res['new_high_pct'], color='#fbbf24', alpha=0.6, label='新高占比')
set_font(ax4, "市场广度与季度强度对比")

# 图5: ETF (修复截断错误)
etfs = {"510050": "上证50", "510300": "沪深300", "510500": "中证500", "512100": "中证1000"}
colors = ['#1e40af', '#166534', '#991b1b', '#6b21a8']
for i, (code, label) in enumerate(etfs.items()):
    ax5.plot(res.index, res[f'turnover_{code}'], label=label, color=colors[i], alpha=0.8)
set_font(ax5, "核心风格 ETF 换手率对比")
if my_font: ax5.legend(prop=my_font, ncol=4)

plt.tight_layout()
st.pyplot(fig)

st.divider()

# --- C. 实战决策报告 (找回并升级你的报告模块) ---
st.subheader("📝 实战决策总结")

latest = res.iloc[-1]
prev = res.iloc[-2]

# 模式判定
if latest['close'] > latest['MA_Filter'] and latest['MA_Filter'] > prev['MA_Filter']:
    mode = "🐂 多头强趋势 (价格站上MA30且均线向上)"
elif latest['close'] < latest['MA_Filter'] and latest['MA_Filter'] < prev['MA_Filter']:
    mode = "🐻 空头弱趋势 (价格跌破MA30且均线向下)"
else:
    mode = "🦓 震荡过渡期 (方向不明，建议减仓观望)"

# 操作建议
signal, pos = latest['signal'], latest['pos']
if signal == 1: 
    action, status = "🚨 立即买入信号", "success"
elif signal == -1: 
    action, status = "🚨 立即卖出信号", "error"
elif pos == 1: 
    action, status = "💎 持股待涨", "info"
else: 
    action, status = "🛡️ 空仓观望", "warning"

# 逻辑扫描
logic_desc = []
if latest['breadth'] < 16: logic_desc.append("📉 广度冰点：全场仅16%个股站上均线，物极必反博弈点")
if latest['Heat_Z'] > 1.5: logic_desc.append("🔥 资金过热：成交量快速放大，需警惕短期风格切换")
if latest['new_high_pct'] > 5: logic_desc.append("💪 内生走强：创60日新高个股比例显著提升")

st.info(f"""
**1. 市场模式**：{mode}  
**2. 资金热度**：{latest['Heat_Z']:.2f} (20日 Z-Score)  
**3. 市场状态**：广度 {latest['breadth']:.2f}% | 60日新高比例 {latest['new_high_pct']:.2f}%  
**4. 操作建议**：{action}  
**5. 逻辑扫描**：{', '.join(logic_desc) if logic_desc else '目前处于常规波动区间'}
""")
