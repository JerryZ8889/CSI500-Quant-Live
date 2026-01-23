import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 网页基础配置与核心参数 (本地字体强力版)
# ==========================================
st.set_page_config(page_title="中证500量化实战决策中心", layout="wide")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# --- 核心修复：直接加载本地字体文件 ---
# 假设你把字体文件传到了 csi500_data 文件夹，文件名必须完全一致！
font_path = './csi500_data/NotoSerifCJKsc-Regular.otf' 

# 检查字体文件是否真的存在
if os.path.exists(font_path):
    # 1. 将字体文件加入 matplotlib 的管理器
    fm.fontManager.addfont(font_path)
    # 2. 获取该字体的内部名称
    custom_font = fm.FontProperties(fname=font_path)
    font_name = custom_font.get_name()
    # 3. 强制设置为全局默认字体
    plt.rcParams['font.family'] = font_name
    # print(f"成功加载字体: {font_name}") # 调试用
else:
    # 保底方案（如果文件没传对，还是尝试系统字体）
    st.error(f"⚠️ 未找到字体文件：{font_path}，请检查路径！")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# --- 修复结束 ---

BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2026-01-15"
MA_FILTER_WINDOW = 30
HEAT_WINDOW = 20

# ==========================================
# 2. 数据整合加载 (路径已针对GitHub结构优化)
# ==========================================
@st.cache_data
def load_data():
    # 路径确保指向你的GitHub子文件夹
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

# ==========================================
# 3. 仿真引擎 (核心逻辑完全保留)
# ==========================================
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
        curr_close, prev_close, curr_ma30 = temp['close'].iloc[i], temp['close'].iloc[i-1], temp['MA_Filter'].iloc[i]
        
        if in_pos:
            if logic_state == "FirstNeg" and cond_comp_b.iloc[i]: logic_state = "Composite"
            exit_f = False
            is_1d, is_below_ma = (curr_close < prev_close), (curr_close < curr_ma30)
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
            elif cond_fn_b_base.iloc[i] and (curr_close > curr_ma30): logic_state, buy_trig = "FirstNeg", True
            if buy_trig:
                temp.iloc[i, temp.columns.get_loc('pos')], temp.iloc[i, temp.columns.get_loc('signal')] = 1, 1
                in_pos, entry_idx, entry_high = True, i, temp['high'].iloc[i]

    actual_pos = temp['pos'].shift(1).fillna(0)
    temp['strat_ret'] = actual_pos * temp['close'].pct_change().fillna(0) - np.where(actual_pos.diff() != 0, 0.001, 0)
    temp['cum_ret'] = (1 + temp['strat_ret']).cumprod()
    return temp

# 数据加载与运行
df_input = load_data()
res = run_strategy(df_input)
res_bench = (1 + df_input['close'].pct_change().fillna(0)).cumprod()

# ==========================================
# 4. 网页布局展示 (指标对齐统计)
# ==========================================
st.title("🛡️ 中证500量化实战决策看板")

st.subheader("📊 策略绩效统计")
cols = st.columns(2)
def get_stats(cum_series):
    total = (cum_series.iloc[-1] - 1) * 100
    mdd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min() * 100
    days = (cum_series.index[-1] - cum_series.index[0]).days
    ann = ((cum_series.iloc[-1])**(365.25/days) - 1) * 100 if days > 0 else 0
    return total, ann, mdd

s_tot, s_ann, s_mdd = get_stats(res['cum_ret'])
b_tot, b_ann, b_mdd = get_stats(res_bench)

with cols[0]:
    st.metric("策略累计收益", f"{s_tot:.2f}%", f"年化: {s_ann:.2f}%")
    st.write(f"策略最大回撤: {s_mdd:.2f}%")
with cols[1]:
    st.metric("中证500基准收益", f"{b_tot:.2f}%", f"年化: {b_ann:.2f}%", delta_color="inverse")
    st.write(f"基准最大回撤: {b_mdd:.2f}%")

st.divider()

# B. 五图联动可视化 (对齐代码A风格)
st.subheader("📈 全维度数据视图")
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 30), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 0.8, 0.8, 1.2, 1.2]})
# 图1: 收益与买卖点
ax1.plot(res_bench, label='中证500基准', color='gray', alpha=0.3, linestyle='--')
ax1.plot(res['cum_ret'], label='MA30同步版策略', color='crimson', linewidth=2)
for sig, col, mark in [(1, 'red', '^'), (-1, 'green', 'v')]:
    pts = res[res['signal'] == sig]
    ax1.scatter(pts.index, res.loc[pts.index, 'cum_ret'], color=col, marker=mark, s=150, zorder=5)
ax1.set_title("策略绩效与实战信号分布", fontsize=15); ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.2)
# 图2: 广度
ax2.plot(res.index, res['breadth'], color='orange', label='MA20上方占比 (%)')
ax2.fill_between(res.index, 0, 100, where=(res['pos']==1), color='blue', alpha=0.1)
ax2.set_title("市场广度波动环境", fontsize=12); ax2.set_ylim(0, 100); ax2.grid(True, alpha=0.2)
# 图3: 热度
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']>=0), color='red', alpha=0.4)
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']<0), color='blue', alpha=0.4)
ax3.axhline(y=1.5, color='darkorange', linestyle='--', label='过热线')
ax3.set_title("资金热度 (20日 Z-Score)", fontsize=12); ax3.legend(loc='upper left')
# 图4: 趋势双轴
ax4_left = ax4; ax4_right = ax4.twinx()
ax4_left.plot(res.index, res['breadth'], color='#1f77b4', linewidth=1.8, label='站上MA20占比')
ax4_right.bar(res.index, res['new_high_pct'], color='sandybrown', alpha=0.6, width=0.8, label='60日新高占比')
ax4_left.set_title("市场广度与季度强度趋势对比", fontsize=12); ax4_left.legend(loc='upper left'); ax4_right.legend(loc='upper right')
# 图5: ETF对比
colors = ['darkblue', 'green', 'red', 'purple']
etfs = {"510050": "上证50", "510300": "沪深300", "510500": "中证500", "512100": "中证1000"}
for i, (code, label) in enumerate(etfs.items()):
    ax5.plot(res.index, res[f'turnover_{code}'], label=f"{label} 换手率", color=colors[i], alpha=0.8)
ax5.set_title("核心风格 ETF 换手率对比", fontsize=12); ax5.legend(loc='upper left', ncol=4); ax5.grid(True, alpha=0.2)
plt.tight_layout()
st.pyplot(fig) # 重要：网页端必须使用 st.pyplot

st.divider()

# C. 实战决策报告
st.subheader("📝 实战决策总结")
latest = res.iloc[-1]
prev = res.iloc[-2]
# 模式判定
if latest['close'] > latest['MA_Filter'] and latest['MA_Filter'] > prev['MA_Filter']:
    mode = "多头 (价格站上MA30且均线向上)"
elif latest['close'] < latest['MA_Filter'] and latest['MA_Filter'] < prev['MA_Filter']:
    mode = "空头 (价格跌破MA30且均线向下)"
else:
    mode = "震荡 (价格与均线纠缠或方向不明)"
# 提醒逻辑
signal, pos = latest['signal'], latest['pos']
if signal == 1: action = "🚨 买入提醒"
elif signal == -1: action = "🚨 卖出提醒"
elif pos == 1: action = "💎 持股待涨"
else: action = "🛡️ 空仓观望"
# 逻辑描述
logic_desc = []
if latest['breadth'] < 16: logic_desc.append("市场广度处于冰点区")
if latest['Heat_Z'] > 1.5: logic_desc.append("资金热度过高")
if latest['new_high_pct'] > 5: logic_desc.append("新高占比显著提升")
st.info(f"""
**1. 市场模式**：{mode}  
**2. 资金热度**：{latest['Heat_Z']:.2f} (20日 Z-Score)  
**3. 市场状态**：广度 {latest['breadth']:.2f}% | 60日新高比例 {latest['new_high_pct']:.2f}%  
**4. 操作建议**：{action}  
**5. 逻辑说明**：{', '.join(logic_desc) if logic_desc else '目前处于常规波动区间'}
""")
