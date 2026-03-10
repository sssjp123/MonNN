import pandas as pd
import os
import numpy as np
import json
import glob
import warnings
from scipy.stats import wilcoxon, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
from scikit_posthocs import critical_difference_diagram
import seaborn as sns

# =====================================================
# 全局设置：字体、路径与警告过滤
# =====================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =====================================================
# 1. 修改后的读取逻辑：适配分散的 Lambda 文件
# =====================================================
def read_csv_files(file_list, directory):
    data_dict = {}
    methods_to_process = {
        'expsMLP': 'exps_MLP.csv',
        'expsPWL': 'exps_PWL_lambda_10.*.csv',
        'expsMixupPWL': 'exps_MixupPWL_lambda_10.*.csv',
        'expsUniformPWL': 'exps_UniformPWL_lambda_10.*.csv'
    }

    for method_key, pattern in methods_to_process.items():
        if method_key == 'expsMLP':
            full_path = os.path.join(directory, pattern)
            if os.path.exists(full_path):
                df = pd.read_csv(full_path)
                df = df.rename(columns={'Metric Mean': 'Metric Value', 'Metric Std': 'Metric Std Dev'})
                data_dict[method_key] = df
        else:
            search_pattern = os.path.join(directory, pattern)
            matching_files = glob.glob(search_pattern)
            if matching_files:
                combined_df = pd.concat([pd.read_csv(f) for f in matching_files])
                combined_df = combined_df.rename(
                    columns={'Metric Mean': 'Metric Value', 'Metric Std': 'Metric Std Dev'})
                data_dict[method_key] = combined_df
    return data_dict


# =====================================================
# 2. 数据处理辅助函数
# =====================================================
def safe_float_convert(value):
    try:
        return float(value)
    except:
        return value


def extract_monotonicity_weight(config_str):
    try:
        config = json.loads(config_str.replace("'", '"'))
        return config.get('monotonicity_weight', 'N/A')
    except:
        return 'N/A'


def create_performance_df(data_dict):
    performance_data = []
    for method, df in data_dict.items():
        for _, row in df.iterrows():
            weight = extract_monotonicity_weight(row['Best Configuration'])
            performance_data.append({
                'Dataset': row['Dataset'], 'Method': method, 'Weight': weight,
                'Metric Value': safe_float_convert(row['Metric Value']),
                'Metric Std Dev': safe_float_convert(row['Metric Std Dev'])
            })
    return pd.DataFrame(performance_data)


def create_mono_dfs(data_dict):
    mono_metrics = ['Mono Random', 'Mono Train', 'Mono Val']
    mono_data = {metric: [] for metric in mono_metrics}
    for method, df in data_dict.items():
        for _, row in df.iterrows():
            weight = extract_monotonicity_weight(row['Best Configuration'])
            for metric in mono_metrics:
                mono_data[metric].append({
                    'Dataset': row['Dataset'], 'Method': method, 'Weight': weight,
                    'Metric Value': safe_float_convert(row[f'{metric} Mean']),
                    'Metric Std Dev': safe_float_convert(row[f'{metric} Std'])
                })
    return {metric: pd.DataFrame(data) for metric, data in mono_data.items()}


# =====================================================
# 3. LaTeX 表格生成逻辑 (Step 2 核心)
# =====================================================
def format_value_with_std(value, std, include_std=True):
    if pd.isna(value): return "N/A"
    if include_std: return f"{value:.4f} $\\pm$ {std:.4f}"
    return f"{value:.4f}"


def df_to_latex(df, caption, label, bold_func, include_std=False):
    methods = sorted(df['Method'].unique())
    weights = sorted([w for w in df['Weight'].unique() if w != 'N/A'])
    datasets = sorted(df.groupby('Dataset').filter(lambda x: len(x['Method'].unique()) > 1)['Dataset'].unique())

    latex_lines = ["\\begin{table}", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                   "\\resizebox{\\textwidth}{!}{"]
    latex_lines.append(
        "\\begin{tabular}{l" + "c" * (len(methods) * (1 if 'expsMLP' in methods else len(weights))) + "}")
    latex_lines.append("\\toprule")

    header_top = ["\\multirow{2}{*}{Dataset}"]
    for method in methods:
        col_span = 1 if method == 'expsMLP' else len(weights)
        header_top.append(f"\\multicolumn{{{col_span}}}{{c}}{{{method}}}")
    latex_lines.append(" & ".join(header_top) + " \\\\")

    header_bottom = [""]
    for method in methods:
        if method == 'expsMLP':
            header_bottom.append("")
        else:
            header_bottom.extend([f"w={w}" for w in weights])
    latex_lines.append(" & ".join(header_bottom) + " \\\\")
    latex_lines.append("\\midrule")

    for dataset in datasets:
        values = [dataset]
        for method in methods:
            m_df = df[df['Dataset'] == dataset]
            if method == 'expsMLP':
                m_d = m_df[m_df['Method'] == method]
                if not m_d.empty:
                    val = format_value_with_std(m_d['Metric Value'].iloc[0], m_d['Metric Std Dev'].iloc[0], include_std)
                    values.append(bold_func(m_df, method, val))
                else:
                    values.append("--")
            else:
                for weight in weights:
                    m_d = m_df[(m_df['Method'] == method) & (m_df['Weight'] == weight)]
                    if not m_d.empty:
                        val = format_value_with_std(m_d['Metric Value'].iloc[0], m_d['Metric Std Dev'].iloc[0],
                                                    include_std)
                        values.append(bold_func(m_df, method, val))
                    else:
                        values.append("--")
        latex_lines.append(" & ".join(values) + " \\\\")

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(latex_lines)


def bold_min_value(df, method, value):
    try:
        val_numeric = float(str(value).split()[0])
        return f"\\textbf{{{value}}}" if val_numeric == df['Metric Value'].min() else value
    except:
        return value


# =====================================================
# 4. 统计检验与支配分析 (核心复现逻辑)
# =====================================================
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def dominance_analysis(df, methods, datasets, weights):
    dominance_counts = {m: 0 for m in methods}
    valid_weights = [w for w in weights if w != 'N/A']
    total_comparisons = len(datasets) * len(valid_weights)

    for dataset in datasets:
        for weight in valid_weights:
            # 关键点：排除 MLP，仅在具有相同 lambda 的正则化方法间对比
            curr = df[(df['Dataset'] == dataset) & (df['Weight'] == weight) & (df['Method'] != 'expsMLP')]
            if not curr.empty:
                costs = curr[['Performance', 'Monotonicity']].values
                pareto = is_pareto_efficient(costs)
                for m, is_opt in zip(curr['Method'], pareto):
                    if is_opt: dominance_counts[m] += 1
    return {m: (c / total_comparisons * 100) for m, c in dominance_counts.items()}


def perform_statistical_tests(df, metric_name):
    print(f"\nPerforming statistical tests for {metric_name}...")
    datasets, methods = df['Dataset'].unique(), df['Method'].unique()
    data = {m: [] for m in methods}
    for d in datasets:
        for m in methods:
            m_d = df[(df['Dataset'] == d) & (df['Method'] == m)]
            data[m].append(m_d['Metric Value'].min() if not m_d.empty else np.nan)
    data = {m: [v for v in vals if not np.isnan(v)] for m, vals in data.items()}
    min_len = min(len(v) for v in data.values())
    data = {m: v[:min_len] for m, v in data.items()}
    try:
        chi2, p = friedmanchisquare(*data.values())
        print(f"Friedman test p-value: {p:.4f}")
        if p < 0.05:
            nemenyi = posthoc_nemenyi_friedman(pd.DataFrame(data))
            print("Nemenyi results:\n", nemenyi)
    except Exception as e:
        print(f"Friedman Error: {e}")

METHOD_LABEL_MAP = {
    "expsMLP": "MLP",
    "expsPWL": "PWL",
    "expsMixupPWL": "MixupPWL",
    "expsUniformPWL": "UniformPWL",
}

def pretty_label(method: str) -> str:
    # Fallback: just remove leading 'exps' if exists
    return METHOD_LABEL_MAP.get(method, method.replace("exps", "", 1))

# =====================================================
# 5. 绘图逻辑 (Times New Roman & 本地保存)
# =====================================================
def plot_performance_monotonicity_tradeoff(performance_df, mono_df):
    combined_df = pd.merge(performance_df, mono_df, on=['Dataset', 'Method', 'Weight'])
    combined_df = combined_df.rename(columns={'Metric Value_x': 'Performance', 'Metric Value_y': 'Monotonicity'})
    plt.figure(figsize=(10, 6))
    for method in combined_df['Method'].unique():
        m_d = combined_df[combined_df['Method'] == method]
        plt.scatter(m_d['Performance'], m_d['Monotonicity'], label=pretty_label(method), alpha=0.6, s=50)
    plt.xlabel('Prediction error', fontsize=12)
    plt.ylabel('Monotonicity violation rate', fontsize=12)
    plt.legend();
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'tradeoff.png'), dpi=300);
    plt.show()


def plot_combined_regularization_impact(performance_df, mono_dfs, chosen_dataset):
    combined_df = performance_df[performance_df['Dataset'] == chosen_dataset].copy()
    for metric, df in mono_dfs.items():
        combined_df = pd.merge(combined_df, df[df['Dataset'] == chosen_dataset], on=['Dataset', 'Method', 'Weight'],
                               suffixes=('', f'_{metric}'))
    combined_df = combined_df.rename(columns={'Metric Value': 'Performance'})

    fig, axes = plt.subplots(4, 1, figsize=(10, 18), sharex=True)
    metrics = ['Performance', 'Mono Random', 'Mono Train', 'Mono Val']
    for ax, m_name in zip(axes, metrics):
        for method in combined_df['Method'].unique():
            if method == 'expsMLP': continue
            m_d = combined_df[combined_df['Method'] == method].copy()
            m_d.loc[:, 'Weight'] = pd.to_numeric(m_d['Weight'], errors='coerce')
            m_d = m_d.sort_values('Weight')
            y_vals = m_d['Performance'] if m_name == 'Performance' else m_d[f'Metric Value_{m_name}']
            ax.plot(m_d['Weight'], y_vals, marker='o', label=pretty_label(method), linewidth=1.5)
        ax.set_xscale('log');
        ax.set_title(f'Impact on {m_name} ({chosen_dataset})', fontsize=14);
        ax.legend();
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f'impact_{chosen_dataset}.png'), dpi=300);
    plt.show()


# =====================================================
# 6. Main 主函数
# =====================================================
def main():
    csv_directory = r'D:\PythonCode\PaperCode\MonNN\src\exps'

    print("--- Step 1: 读取并合并 Lambda 汇总 CSV 文件 ---")
    data_dict = read_csv_files([], csv_directory)
    if not data_dict:
        print("Error: No CSV files found!");
        return

    performance_df = create_performance_df(data_dict)
    mono_dfs = create_mono_dfs(data_dict)

    print("\n--- Step 2: 生成 Performance LaTeX 表格 ---")
    print(df_to_latex(performance_df, "Performance Metrics", "tab:perf", bold_min_value))

    print("\n--- Step 3: 支配分析 (MLP 锁定为 0.00%) ---")
    for metric, df in mono_dfs.items():
        combined_df = pd.merge(performance_df, df, on=['Dataset', 'Method', 'Weight'])
        combined_df = combined_df.rename(columns={'Metric Value_x': 'Performance', 'Metric Value_y': 'Monotonicity'})

        results = dominance_analysis(combined_df, combined_df['Method'].unique(), combined_df['Dataset'].unique(),
                                     combined_df['Weight'].unique())
        print(f"\nDominance Analysis for {metric}:")
        for m, p in results.items(): print(f"{m}: {p:.2f}% Pareto-optimal")

    print("\n--- Step 4: 统计检验 ---")
    perform_statistical_tests(performance_df, "Performance Metrics")

    print("\n--- Step 5: 绘图完成并保存本地 ---")
    if not performance_df.empty:
        example_ds = performance_df['Dataset'].unique()[0]
        plot_combined_regularization_impact(performance_df, mono_dfs, example_ds)
        plot_performance_monotonicity_tradeoff(performance_df, list(mono_dfs.values())[0])

    print(f"\n任务结束。图片已保存至: {BASE_DIR}")


if __name__ == "__main__":
    main()