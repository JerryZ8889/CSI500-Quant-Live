import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil

# ==========================================
# 1. 网页基础配置
# ==========================================
st.set_page_config(page_title="中证500量化实战决策中心", layout="wide")

# 清除缓存，确保重载字体
cache_dir = os.path.expanduser('~/.cache/matplotlib')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)

# 字体路径（确保你的 SimHei.ttf 在这个位置）
font_path = './csi500_data/SimHei.ttf'

# ==========================================
# 2. 数据处理与仿真逻辑 (逻辑保持不变)
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
            df[f'turnover_{code}'] = np.where(df[f'turn_raw_{code}'].max() > 1, df[f'turn_raw_{code}'], df[f'turn_raw_{code}'] * 100)
        else:
            df[f'turnover_{code}'] = 0
    df['new_high_pct'] = df['new_high_pct'].fillna(0)
    df['MA_Filter'] = df['close'].rolling(MA_FILTER_WINDOW).mean() 
    df['Heat_Z'] = (df['amount'] - df['amount'].rolling(HEAT_WINDOW).mean()) / df['amount'].rolling(HEAT_WINDOW).std()
    df['Turnover_Pct'] = np.where(df['ETF_Turnover'].max() > 1, df['ETF_Turnover'], df['ETF_Turnover'] * 100)
    # 策略计算中需要的辅助列
    df['MA_Trend'] = df['close'].rolling(10).mean()
    df['MA_Support'] = df['close'].rolling(5).mean()
    df['Is_Up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['Streak'] = df['Is_Up'].groupby((df['Is_Up'] != df['Is_Up'].shift()).cumsum()).cumcount() + 1
    df['Consec_Gains'] = np.where(df['Is_Up'] == 1, df['Streak'], 0)
    return df.sort_values('date').set_index('date').loc[BACKTEST_START:BACKTEST_END]

def run_strategy(df_main):
    temp = df_main.copy()
    temp['pos'], temp['signal'] = 0, 0
    in_pos, logic_state, entry_idx, entry_high = False, "", 0, 0
    cond_comp_b = (temp['breadth'] < 16)
    cond_comp_s = (temp['breadth'] > 79) & (temp['Heat_Z'] < 1.5)
    cond_fn_b_base = (temp['close'] > temp['MA_Trend']) & (temp['Consec_Gains'].shift(1) >= 3) & (temp['close'] < temp['close'].shift(1)) & (temp['Turnover_Pct'] > 1.0) & (temp['close'] > temp['MA_Support'])
    for i in range(len(temp)):
        if i == 0: continue
        curr_c, prev_c, ma30 = temp['close'].iloc[i], temp['close'].iloc[i-1], temp['MA_Filter'].iloc[i]
        if in_pos:
            if logic_state == "FirstNeg" and cond_comp_b.iloc[i]: logic_state = "Composite"
            exit_f = False
            if logic_state == "Composite":
                if cond_comp_s.iloc[i]: exit_f = True
            else:
                if cond_comp_s.iloc[i]: exit_f = True
                elif (curr_c < ma30) and ((curr_c < prev_c) or (i - entry_idx >= 5 and not (temp['close'].iloc[entry_idx:i+1] > entry_high).any())): exit_f = True
            if exit_f:
                temp.iloc[i, temp.columns.get_loc('pos')], temp.iloc[i, temp.columns.get_loc('signal')] = 0, -1
                in_pos = False
            else: temp.iloc[i, temp.columns.get_loc('pos')] = 1
        else:
            buy_trig = False
            if cond_comp_b.iloc[i]: logic_state, buy_trig = "Composite", True
            elif cond_fn_b_base.iloc[i] and (curr_c > ma30): logic_state, buy_trig = "FirstNeg", True
            if buy_trig:
                temp.iloc[i, temp.columns.get_loc('pos')], temp.iloc[i, temp.columns.get_loc('signal')] = 1, 1
                in_pos, entry_idx, entry_high = True, i, temp['high'].iloc[i]
    temp['strat_ret'] = temp['pos'].shift(1).fillna(0) * temp['close'].pct_change().fillna(0) - np.where(temp['pos'].shift(1).diff() != 0, 0.001, 0)
    temp['cum_ret'] = (1 + temp['strat_ret']).cumprod()
    return temp

# 数据准备
df_input = load_data()
res = run_strategy(df_input)
res_bench = (1 + df_input['close'].pct_change().fillna(0)).cumprod()

# ==========================================
# 3. 网页布局与绘图 (重点修复区)
# ==========================================
st.title("🛡️ 中证500 | 量化实战决策看板")

# --- 绩效统计卡片 ---
s_tot = (res['cum_ret'].iloc[-1]-1)*100
b_tot = (res_bench.iloc[-1]-1)*100
col1, col2, col3, col4 = st.columns(4)
col1.metric("🚀 策略累计收益", f"{s_tot:.2f}%")
col2.metric("📉 策略最大回撤", f"{((res['cum_ret'] - res['cum_ret'].cummax()) / res['cum_ret'].cummax()).min() * 100:.2f}%")
col3.metric("🏛️ 基准累计收益", f"{b_tot:.2f}%")
col4.metric("🌊 基准最大回撤", f"{((res_bench - res_bench.cummax()) / res_bench.cummax()).min() * 100:.2f}%")

st.divider()

# --- 绘图配置 (解决乱码的关键) ---
# 必须先执行 style.use
plt.style.use('seaborn-v0_8-whitegrid')

# 获取字体属性对象 (双重保险)
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    # 获取字体文件的内部名称
    my_font_name = fm.FontProperties(fname=font_path).get_name()
    # 强制设为全局默认
    plt.rcParams['font.family'] = my_font_name
    plt.rcParams['font.sans-serif'] = [my_font_name]
    plt.rcParams['axes.unicode_minus'] = False
    # 创建属性对象用于局部注入
    fprop = fm.FontProperties(fname=font_path)
else:
    fprop = None
    st.warning("字体文件未找到，图表可能显示异常")

# 开始绘图
st.subheader("📈 全维度数据视图")
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 25), sharex=True, gridspec_kw={'height_ratios': [2, 0.8, 0.8, 1, 1]})

# 图1：收益曲线
ax1.plot(res_bench, label='中证500基准', color='gray', alpha=0.3, linestyle='--')
ax1.plot(res['cum_ret'], label='MA30同步版策略', color='crimson', linewidth=2)
for sig, col, mark in [(1, 'red', '^'), (-1, 'green', 'v')]:
    pts = res[res['signal'] == sig]
    ax1.scatter(pts.index, res.loc[pts.index, 'cum_ret'], color=col, marker=mark, s=150, zorder=5)
# 在 set_title 和 legend 中强制注入字体 (局部保险)
ax1.set_title("策略绩效与实战信号分布", fontproperties=fprop, fontsize=16)
ax1.legend(prop=fprop, loc='upper left')

# 图2：市场广度
ax2.plot(res.index, res['breadth'], color='orange', label='MA20上方占比 (%)')
ax2.fill_between(res.index, 0, 100, where=(res['pos']==1), color='blue', alpha=0.05)
ax2.set_title("市场广度波动环境", fontproperties=fprop); ax2.set_ylim(0, 100)

# 图3：资金热度
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']>=0), color='red', alpha=0.4)
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']<0), color='blue', alpha=0.4)
ax3.axhline(y=1.5, color='darkorange', linestyle='--', label='过热线')
ax3.set_title("资金热度 (20日 Z-Score)", fontproperties=fprop)

# 图4：广度与强度
ax4_right = ax4.twinx()
ax4.plot(res.index, res['breadth'], color='#1f77b4', label='站上MA20占比')
ax4_right.bar(res.index, res['new_high_pct'], color='sandybrown', alpha=0.6, label='60日新高占比')
ax4.set_title("市场广度与季度强度对比", fontproperties=fprop)

# 图5：ETF换手率对比 (修复之前的截断报错)
colors = ['darkblue', 'green', 'red', 'purple']
etfs = {"510050": "上证50", "510300": "沪深300", "510500": "中证500", "512100": "中证1000"}
for i, (code, label) in enumerate(etfs.items()):
    ax5.plot(res.index, res[f'turnover_{code}'], label=label, color=colors[i], alpha=0.8)
ax5.set_title("核心风格 ETF 换手率对比", fontproperties=fprop)
ax5.legend(prop=fprop, loc='upper left', ncol=4)

plt.tight_layout()
st.pyplot(fig)
