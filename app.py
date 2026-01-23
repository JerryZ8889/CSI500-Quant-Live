import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ==========================================
# 1. 网页基础配置与视觉优化
# ==========================================
st.set_page_config(
    page_title="中证500量化实战决策中心", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 注入自定义 CSS (美化关键) ---
st.markdown("""
    <style>
    /* 全局字体优化 */
    .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* 指标卡片样式 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 核心修复：更稳健的字体加载逻辑 ---
# 建议上传 SimHei.ttf 到 csi500_data 文件夹，它的兼容性最好
font_path = './csi500_data/SimHei.ttf' 

# 使用 try-except 防止因为字体文件坏了导致整个 App 崩溃
try:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        custom_font = fm.FontProperties(fname=font_path)
        font_name = custom_font.get_name()
        plt.rcParams['font.family'] = font_name
        # print(f"成功加载本地字体: {font_name}")
    else:
        # 如果找不到文件，静默回退，不要报错
        print(f"提示：未在 {font_path} 找到字体文件，尝试使用系统默认字体。")
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
except Exception as e:
    # 万一字体文件损坏，捕获异常，保证程序能跑
    print(f"字体加载异常 (已忽略): {e}")
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False
# ----------------------------------------

BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2026-01-15"
MA_FILTER_WINDOW = 30
HEAT_WINDOW = 20

# ==========================================
# 2. 数据整合加载 (逻辑保持不变)
# ==========================================
@st.cache_data
def load_data():
    path_prefix = "./csi500_data/"
    # 简单的容错：如果文件不存在，提示用户
    if not os.path.exists(f"{path_prefix}sh.000905.csv"):
        st.error("❌ 数据文件未找到！请确保 sh.000905.csv 等文件已上传到 csi500_data 目录。")
        return pd.DataFrame()

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
# 3. 仿真引擎 (逻辑保持不变)
# ==========================================
def run_strategy(df_main):
    if df_main.empty: return pd.DataFrame()
    
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

# 数据计算
df_input = load_data()
if df_input.empty:
    st.stop() # 数据没加载成功就停止渲染

res = run_strategy(df_input)
res_bench = (1 + df_input['close'].pct_change().fillna(0)).cumprod()

# ==========================================
# 4. 网页布局展示 (UI 美化版)
# ==========================================

# 4.1 头部标题区
st.markdown("## 🛡️ 中证500 | 量化实战决策看板")
st.markdown("---")

# 4.2 绩效卡片区
st.subheader("📊 策略绩效统计")

def get_stats(cum_series):
    total = (cum_series.iloc[-1] - 1) * 100
    mdd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min() * 100
    days = (cum_series.index[-1] - cum_series.index[0]).days
    ann = ((cum_series.iloc[-1])**(365.25/days) - 1) * 100 if days > 0 else 0
    return total, ann, mdd

s_tot, s_ann, s_mdd = get_stats(res['cum_ret'])
b_tot, b_ann, b_mdd = get_stats(res_bench)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="🚀 策略累计收益", value=f"{s_tot:.2f}%", delta=f"年化 {s_ann:.2f}%")
with col2:
    st.metric(label="📉 策略最大回撤", value=f"{s_mdd:.2f}%", delta_color="off")
with col3:
    st.metric(label="🏛️ 基准累计收益", value=f"{b_tot:.2f}%", delta=f"年化 {b_ann:.2f}%")
with col4:
    st.metric(label="🌊 基准最大回撤", value=f"{b_mdd:.2f}%", delta_color="off")

st.markdown("---")

# 4.3 核心图表区 (美化版)
st.subheader("📈 全维度数据视图")

# 设置图表风格
plt.style.use('seaborn-v0_8-whitegrid')

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 30), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 0.8, 0.8, 1.2, 1.2]})

# 统一背景色
fig.patch.set_facecolor('none')
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_facecolor('none')
    ax.tick_params(axis='both', which='major', labelsize=10)

# 图1: 收益与买卖点
ax1.plot(res_bench, label='中证500基准', color='#95a5a6', alpha=0.5, linestyle='--', linewidth=1.5)
ax1.plot(res['cum_ret'], label='MA30同步版策略', color='#c0392b', linewidth=2.5) 
for sig, col, mark in [(1, '#e74c3c', '^'), (-1, '#27ae60', 'v')]:
    pts = res[res['signal'] == sig]
    ax1.scatter(pts.index, res.loc[pts.index, 'cum_ret'], color=col, marker=mark, s=180, zorder=5, edgecolors='white', linewidth=1.5)
ax1.set_title("策略绩效与实战信号分布", fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# 图2: 广度
ax2.plot(res.index, res['breadth'], color='#f39c12', label='MA20上方占比 (%)', linewidth=1.5)
ax2.fill_between(res.index, 0, 100, where=(res['pos']==1), color='#3498db', alpha=0.1)
ax2.set_title("市场广度波动环境", fontsize=14, pad=10)
ax2.set_ylim(0, 100)
ax2.set_ylabel("占比 %")

# 图3: 热度
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']>=0), color='#e74c3c', alpha=0.5, label='资金流入')
ax3.fill_between(res.index, 0, res['Heat_Z'], where=(res['Heat_Z']<0), color='#2980b9', alpha=0.5, label='资金流出')
ax3.axhline(y=1.5, color='#d35400', linestyle='--', linewidth=1.5, label='过热警戒线 (1.5)')
ax3.set_title("资金热度 (20日 Z-Score)", fontsize=14, pad=10)
ax3.legend(loc='upper left', fontsize=9)

# 图4: 趋势双轴
ax4_left = ax4; ax4_right = ax4.twinx()
ax4_left.plot(res.index, res['breadth'], color='#2980b9', linewidth=2, label='站上MA20占比')
ax4_right.bar(res.index, res['new_high_pct'], color='#e67e22', alpha=0.5, width=1.0, label='60日新高占比')
ax4_left.set_title("市场广度与季度强度趋势对比", fontsize=14, pad=10)
ax4_left.legend(loc='upper left', fontsize=9)
ax4_right.legend(loc='upper right', fontsize=9)
ax4_right.set_ylabel("新高占比 %")

# 图5: ETF对比
colors = ['#2c3e50', '#27ae60', '#c0392b', '#8e44ad']
etfs = {"510050": "上证50", "510300": "沪深300", "510500": "中证500", "512100": "中证1000"}
for i, (code, label) in enumerate(etfs.items()):
    ax5.plot(res.index, res[f'turnover_{code}'], label=f"{label}", color=colors[i], alpha=0.8, linewidth=1.5)
ax5.set_title("核心风格 ETF 换手率对比", fontsize=14, pad=10)
ax5.legend(loc='upper left', ncol=4, fontsize=10)

plt.tight_layout()
st.pyplot(fig)

st.divider()

# 4.4 决策看板 (布局重构)
st.subheader("📝 实战决策总结")

latest = res.iloc[-1]
prev = res.iloc[-2]

# 模式判定
if latest['close'] > latest['MA_Filter'] and latest['MA_Filter'] > prev['MA_Filter']:
    mode = "🐂 多头趋势"
    mode_desc = "价格站上MA30且均线向上"
elif latest['close'] < latest['MA_Filter'] and latest['MA_Filter'] < prev['MA_Filter']:
    mode = "🐻 空头趋势"
    mode_desc = "价格跌破MA30且均线向下"
else:
    mode = "🦓 震荡整理"
    mode_desc = "价格与均线纠缠或方向不明"

# 提醒逻辑
signal, pos = latest['signal'], latest['pos']
if signal == 1: 
    action = "🚨 买入信号"
    action_type = "success"
elif signal == -1: 
    action = "🚨 卖出信号"
    action_type = "error"
elif pos == 1: 
    action = "💎 持股待涨"
    action_type = "info"
else: 
    action = "🛡️ 空仓观望"
    action_type = "secondary"

logic_desc = []
if latest['breadth'] < 16: logic_desc.append("📉 广度冰点")
if latest['Heat_Z'] > 1.5: logic_desc.append("🔥 资金过热")
if latest['new_high_pct'] > 5: logic_desc.append("💪 新高增强")
final_logic = " | ".join(logic_desc) if logic_desc else "🌊 市场处于常规波动区间"

with st.container():
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1.5])
    
    with c1:
        st.markdown(f"#### 1. 市场模式")
        st.markdown(f"**{mode}**")
        st.caption(mode_desc)
    with c2:
        st.markdown(f"#### 2. 资金热度")
        st.metric("Z-Score", f"{latest['Heat_Z']:.2f}", delta=None)
    with c3:
        st.markdown(f"#### 3. 市场结构")
        st.metric("广度 / 新高", f"{latest['breadth']:.0f}%", delta=f"{latest['new_high_pct']:.1f}% 新高")
    with c4:
        st.markdown(f"#### 4. 操作建议")
        if action_type == "success": st.success(f"### {action}")
        elif action_type == "error": st.error(f"### {action}")
        elif action_type == "info": st.info(f"### {action}")
        else: st.info(f"### {action}") # 这里的 secondary 某些旧版不支持，改回 info 保底

    st.info(f"**逻辑扫描：** {final_logic}")
