import pandas as pd
import os
import numpy as np
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
from scikit_posthocs import critical_difference_diagram
import ast

# 读取 CSV 文件并进行数据清理
def read_csv_files(file_list, directory):
    data_dict = {}
    for file in file_list:
        method_name = os.path.splitext(file)[0]
        full_path = os.path.join(directory, file)
        df = pd.read_csv(full_path)

        # 清理数据：将所有的 '--' 替换为 NaN
        df.replace('--', np.nan, inplace=True)

        # 将 NaN 转换为数字
        df = df.apply(pd.to_numeric, errors='ignore')

        data_dict[method_name] = df
    return data_dict

# 创建性能数据框
def create_performance_df(data_dict):
    performance_data = {}
    performance_std_data = {}
    for method, df in data_dict.items():
        if 'Metric Mean' in df.columns and 'Metric Std' in df.columns:
            # 确保索引是 Dataset
            performance_data[method] = df.set_index('Dataset')['Metric Mean']
            performance_std_data[method] = df.set_index('Dataset')['Metric Std']
        else:
            print(f"Warning: Metric columns not found for method {method}")

    performance_df = pd.DataFrame(performance_data)
    performance_std_df = pd.DataFrame(performance_std_data)

    # 修改点：不再使用 dropna(axis=1)，而是填充 NaN 或保留，
    # 否则只要 MM 有一个空值，整列 MM 都会被删掉
    # 这里建议只对行进行处理（如果某个数据集在所有方法中都没有数据）
    performance_df = performance_df.dropna(axis=0, how='all')
    performance_std_df = performance_std_df.dropna(axis=0, how='all')

    return performance_df.sort_index(), performance_std_df.sort_index()

# 创建参数数据框
def create_params_df(data_dict):
    params_data = {}
    for method, df in data_dict.items():
        params_data[method] = df.set_index('Dataset')['NumOfParameters']
    params_df = pd.concat(params_data, axis=1)
    return params_df.sort_index()

# 创建单调性数据框
def create_mono_dfs(data_dict):
    mono_metrics = ['Mono Random', 'Mono Train', 'Mono Val']
    mono_dfs = {}
    for metric in mono_metrics:
        mono_data = {}
        mono_std_data = {}
        for method, df in data_dict.items():
            if f'{metric} Mean' in df.columns and f'{metric} Std' in df.columns:
                mono_data[method] = df.set_index('Dataset')[f'{metric} Mean']
                mono_std_data[method] = df.set_index('Dataset')[f'{metric} Std']
            else:
                print(f"Warning: {metric} columns not found for method {method}")

        if mono_data:
            mono_df = pd.concat(mono_data, axis=1)
            mono_std_df = pd.concat(mono_std_data, axis=1)
            mono_dfs[metric] = (mono_df.sort_index(), mono_std_df.sort_index())
        else:
            print(f"Warning: No data for {metric}.")

    return mono_dfs

def format_value_with_std(value, std):
    return f"{value:.4f} $\\pm$ {std:.4f}"

def extract_value(formatted_string):
    return float(formatted_string.split()[0])

def bold_min_value(series):
    values = series.apply(extract_value)
    min_idx = values.idxmin()
    return series.apply(lambda x: f"\\textbf{{{x}}}" if x == series[min_idx] else x)

def bold_max_value(series):
    values = series.apply(extract_value)
    max_idx = values.idxmax()
    return series.apply(lambda x: f"\\textbf{{{x}}}" if x == series[max_idx] else x)

def df_to_latex(df, std_df, caption, label, bold_func):
    formatted_df = pd.DataFrame()
    for method in df.columns:
        formatted_values = df[method].combine(std_df[method], format_value_with_std)
        formatted_df[method] = formatted_values

    for idx in formatted_df.index:
        formatted_df.loc[idx] = bold_func(formatted_df.loc[idx])

    latex_table = formatted_df.to_latex(escape=False, caption=caption, label=label)
    return latex_table

def params_df_to_latex(df, caption, label):
    formatted_df = df.astype(str)
    for idx in formatted_df.index:
        formatted_df.loc[idx] = bold_min_value(formatted_df.loc[idx])

    latex_table = formatted_df.to_latex(escape=False, caption=caption, label=label)
    return latex_table

def perform_wilcoxon_tests(df):
    methods = df.columns
    n_methods = len(methods)
    results = pd.DataFrame(index=methods, columns=methods)

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method1 = methods[i]
            method2 = methods[j]
            try:
                _, p_value = wilcoxon(df[method1], df[method2])
                results.loc[method1, method2] = p_value
                results.loc[method2, method1] = p_value
            except Exception as e:
                print(f"Error performing Wilcoxon test between {method1} and {method2}: {str(e)}")
                results.loc[method1, method2] = np.nan
                results.loc[method2, method1] = np.nan

    return results

def perform_friedman_test(df):
    try:
        chi2, p_value = friedmanchisquare(*[df[col] for col in df.columns])
        return chi2, p_value
    except Exception as e:
        print(f"Error performing Friedman test: {str(e)}")
        return np.nan, np.nan

def perform_nemenyi_test(df):
    try:
        return posthoc_nemenyi_friedman(df)
    except Exception as e:
        print(f"Error performing Nemenyi test: {str(e)}")
        return None

def create_latex_table(df, caption, label):
    latex_table = df.to_latex(float_format="%.4f", caption=caption, label=label, escape=False)
    return latex_table

def create_critical_difference_diagram(performance_df, filename='critical_difference_diagram.png'):
    # Compute average ranks
    ranks = performance_df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    # Prepare data for the diagram
    ranks_dict = avg_ranks.to_dict()

    # Perform Nemenyi test to get the p-value matrix
    nemenyi_results = posthoc_nemenyi_friedman(performance_df)

    # Create the diagram
    fig, ax = plt.subplots(figsize=(10, 5))

    critical_difference_diagram(
        ranks=ranks_dict,
        sig_matrix=nemenyi_results,
        ax=ax,
        label_fmt_left='{label} ({rank:.2f})',
        label_fmt_right='({rank:.2f}) {label}',
    )

    plt.title("Critical Difference Diagram")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Critical Difference Diagram saved as {filename}")


def main():
    csv_directory = r'D:\PythonCode\PaperCode\MonNN\src\exps'
    csv_files = [
        'exps_MM.csv', 'exps_MMaux.csv',
        'exps_SMM.csv', 'exps_SMMaux.csv'
    ]

    data_dict = read_csv_files(csv_files, csv_directory)

    performance_df, performance_std_df = create_performance_df(data_dict)
    params_df = create_params_df(data_dict)

    # 打印检查，确保四列都在
    print("參與分析的方法列表:", performance_df.columns.tolist())

    try:
        performance_latex = df_to_latex(performance_df, performance_std_df, "Performance Metrics", "tab:performance",
                                        bold_min_value)
        print("\nPerformance Table created successfully")
        print(performance_latex)
    except Exception as e:
        print(f"Error creating Performance Table: {str(e)}")

    print("\n--- Starting Statistical Analysis ---")
    print("Performing Friedman test...")
    chi2, p_value = perform_friedman_test(performance_df)
    print(f"Friedman test statistic: {chi2}")
    print(f"Friedman test p-value: {p_value}")

    # --- 强制执行部分，删除了 if p_value < 0.05 ---
    print("\n[Mandatory Execution] Performing Wilcoxon signed-rank tests...")
    wilcoxon_results = perform_wilcoxon_tests(performance_df)
    print("Wilcoxon test results:")
    print(wilcoxon_results)

    print("\n[Mandatory Execution] Performing Nemenyi post-hoc test...")
    nemenyi_results = perform_nemenyi_test(performance_df)
    print("Nemenyi test results:")
    print(nemenyi_results)

    # 转换为 LaTeX
    wilcoxon_latex = create_latex_table(wilcoxon_results, "Wilcoxon Test Results", "tab:wilcoxon")
    nemenyi_latex = create_latex_table(nemenyi_results, "Nemenyi Test Results", "tab:nemenyi")

    print("\nWilcoxon Test Results (LaTeX):")
    print(wilcoxon_latex)
    print("\nNemenyi Test Results (LaTeX):")
    print(nemenyi_latex)

    # 生成 CD 图
    try:
        create_critical_difference_diagram(performance_df, filename='CD_diagram.png')
    except Exception as e:
        print(f"Error generating CD diagram: {e}")


if __name__ == "__main__":
    main()