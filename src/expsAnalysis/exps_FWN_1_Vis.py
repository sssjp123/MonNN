import os
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import wilcoxon, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman, critical_difference_diagram

import matplotlib as mpl
import matplotlib.pyplot as plt


# Global plotting style
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts
mpl.rcParams["ps.fonttype"] = 42



def read_csv_files(file_list, directory):
    data_dict = {}
    for file in file_list:
        method_name = os.path.splitext(file)[0]
        full_path = os.path.join(directory, file)
        df = pd.read_csv(full_path)

        # Clean: replace '--' with NaN
        df.replace("--", np.nan, inplace=True)

        # Convert numeric columns when possible
        df = df.apply(pd.to_numeric, errors="ignore")

        data_dict[method_name] = df
    return data_dict


def create_performance_df(data_dict):
    performance_data = {}
    performance_std_data = {}

    for method, df in data_dict.items():
        if "Metric Mean" in df.columns and "Metric Std" in df.columns:
            performance_data[method] = df.set_index("Dataset")["Metric Mean"]
            performance_std_data[method] = df.set_index("Dataset")["Metric Std"]
        else:
            print(f"Warning: Metric columns not found for method {method}")

    performance_df = pd.concat(performance_data, axis=1)
    performance_std_df = pd.concat(performance_std_data, axis=1)

    performance_df = performance_df.dropna(axis=1, how="any")
    performance_std_df = performance_std_df.dropna(axis=1, how="any")

    return performance_df.sort_index(), performance_std_df.sort_index()


def create_params_df(data_dict):
    params_data = {}
    for method, df in data_dict.items():
        if "NumOfParameters" not in df.columns:
            print(f"Warning: NumOfParameters not found for method {method}")
            continue
        params_data[method] = df.set_index("Dataset")["NumOfParameters"]

    params_df = pd.concat(params_data, axis=1)
    return params_df.sort_index()


def create_mono_dfs(data_dict):
    mono_metrics = ["Mono Random", "Mono Train", "Mono Val"]
    mono_dfs = {}
    for metric in mono_metrics:
        mono_data = {}
        mono_std_data = {}
        for method, df in data_dict.items():
            if f"{metric} Mean" in df.columns and f"{metric} Std" in df.columns:
                mono_data[method] = df.set_index("Dataset")[f"{metric} Mean"]
                mono_std_data[method] = df.set_index("Dataset")[f"{metric} Std"]
            else:
                print(f"Warning: {metric} columns not found for method {method}")

        if mono_data:
            mono_df = pd.concat(mono_data, axis=1)
            mono_std_df = pd.concat(mono_std_data, axis=1)
            mono_dfs[metric] = (mono_df.sort_index(), mono_std_df.sort_index())
        else:
            print(f"Warning: No data for {metric}.")

    return mono_dfs

def prettify_method_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        new_c = c
        if isinstance(new_c, str) and new_c.startswith("exps_"):
            new_c = new_c.replace("exps_", "", 1)
        if new_c == "WeightsConstrained":
            new_c = r"MLP$_{\mathrm{wc}}$"
        rename_map[c] = new_c
    return df.rename(columns=rename_map)

# LaTeX formatting functions
def format_value_with_std(value, std):
    return f"{value:.4f} $\\pm$ {std:.4f}"


def extract_value(formatted_string):
    # formatted: "<mean> $\pm$ <std>"
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
    formatted_df = pd.DataFrame(index=df.index)
    for method in df.columns:
        formatted_values = df[method].combine(std_df[method], format_value_with_std)
        formatted_df[method] = formatted_values

    for idx in formatted_df.index:
        formatted_df.loc[idx] = bold_func(formatted_df.loc[idx])

    latex_table = formatted_df.to_latex(escape=False, caption=caption, label=label)
    return latex_table


def params_df_to_latex(df, caption, label):
    formatted_df = df.copy()
    # ensure string display
    formatted_df = formatted_df.astype(int).astype(str)

    # bold minimum parameter count per dataset
    for idx in formatted_df.index:
        # numeric compare on original df row
        row = df.loc[idx].astype(float)
        min_val = row.min()
        formatted_df.loc[idx] = [
            f"\\textbf{{{int(v)}}}" if float(v) == float(min_val) else f"{int(v)}"
            for v in row.values
        ]

    latex_table = formatted_df.to_latex(escape=False, caption=caption, label=label)
    return latex_table


def create_latex_table(df, caption, label):
    latex_table = df.to_latex(float_format="%.4f", caption=caption, label=label, escape=False)
    return latex_table


# Statistical tests
def perform_wilcoxon_tests(df):
    methods = df.columns
    n_methods = len(methods)
    results = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method1 = methods[i]
            method2 = methods[j]
            try:
                _, p_value = wilcoxon(df[method1], df[method2])
                results.loc[method1, method2] = p_value
                results.loc[method2, method1] = p_value
            except Exception as e:
                print(f"Error performing Wilcoxon test between {method1} and {method2}: {e}")
                results.loc[method1, method2] = np.nan
                results.loc[method2, method1] = np.nan

    np.fill_diagonal(results.values, np.nan)
    return results


def perform_friedman_test(df):
    try:
        chi2, p_value = friedmanchisquare(*[df[col] for col in df.columns])
        return chi2, p_value
    except Exception as e:
        print(f"Error performing Friedman test: {e}")
        return np.nan, np.nan


def perform_nemenyi_test(df):
    try:
        return posthoc_nemenyi_friedman(df)
    except Exception as e:
        print(f"Error performing Nemenyi test: {e}")
        return None



# CD diagram
def create_avg_rank_and_cd_diagram(
    performance_df,
    out_dir=None,
    cd_filename="CD_diagram.png",
    rank_filename="Average_Ranks.png",
    title_cd="Critical Difference Diagram (Nemenyi)",
    title_rank="Average Ranks (lower error is better)",
):
    """
    Assumes performance_df is an error metric: lower is better.
    Saves both plots into out_dir (default: script directory).
    """

    # output directory: script directory by default
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent
    else:
        out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Average ranks: lower is better => ascending=True
    ranks = performance_df.rank(axis=1, ascending=True, method="average")
    avg_ranks = ranks.mean().sort_values()  # smaller rank => better
    ranks_dict = avg_ranks.to_dict()

    # Nemenyi significance matrix
    nemenyi_results = posthoc_nemenyi_friedman(performance_df)

    # Average ranks plot
    fig, ax = plt.subplots(figsize=(8, 4.6))
    avg_ranks.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average rank")
    ax.set_xlabel("Method")
    # ax.set_title(title_rank)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    rank_path = out_dir / rank_filename
    plt.savefig(rank_path, dpi=300, bbox_inches="tight")
    plt.close()

    # CD diagram
    fig, ax = plt.subplots(figsize=(10, 4.8))
    critical_difference_diagram(
        ranks=ranks_dict,
        sig_matrix=nemenyi_results,
        ax=ax,
        label_fmt_left="{label} ({rank:.2f})",
        label_fmt_right="({rank:.2f}) {label}",
    )
    # ax.set_title(title_cd)
    plt.tight_layout()

    cd_path = out_dir / cd_filename
    plt.savefig(cd_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] Average ranks plot -> {rank_path}")
    print(f"[Saved] CD diagram         -> {cd_path}")

    return avg_ranks, nemenyi_results, str(rank_path), str(cd_path)



# Main
def main():
    csv_directory = r"D:\PythonCode\PaperCode\MonNN\src\exps"
    csv_files = [
        "exps_MLP.csv",
        "exps_WeightsConstrained.csv",
        "exps_MM.csv",
        "exps_SMM.csv",
        "exps_HLL.csv",
        "exps_UMNN.csv",
        "exps_LMN.csv",
        "exps_CoMNN.csv",
        "exps_SMNN.csv",
    ]

    data_dict = read_csv_files(csv_files, csv_directory)

    performance_df, performance_std_df = create_performance_df(data_dict)
    params_df = create_params_df(data_dict)
    mono_dfs = create_mono_dfs(data_dict)

    performance_df = prettify_method_names(performance_df)
    performance_std_df = prettify_method_names(performance_std_df)
    params_df = prettify_method_names(params_df)

    mono_dfs = {k: (prettify_method_names(v[0]), prettify_method_names(v[1])) for k, v in mono_dfs.items()}


    # LaTeX tables
    try:
        performance_latex = df_to_latex(
            performance_df,
            performance_std_df,
            "Performance Metrics",
            "tab:performance",
            bold_min_value,  # error: bold min
        )
        print("Performance Table created successfully")
        print(performance_latex)
    except Exception as e:
        print(f"Error creating Performance Table: {e}")

    try:
        params_latex = params_df_to_latex(params_df, "Number of Parameters", "tab:params")
        print("Parameters Table created successfully")
        print(params_latex)
    except Exception as e:
        print(f"Error creating Parameters Table: {e}")


    # Friedman test
    print("Performing Friedman test...")
    chi2, p_value = perform_friedman_test(performance_df)
    print(f"Friedman test statistic: {chi2}")
    print(f"Friedman test p-value: {p_value}")


    # Average ranks and CD diagram
    try:
        avg_ranks, nemenyi_mat, rank_png, cd_png = create_avg_rank_and_cd_diagram(
            performance_df,
            out_dir=None,  # save to script directory
            cd_filename="CD_diagram.png",
            rank_filename="Average_Ranks.png",
            # title_cd="Critical difference diagram",
            # title_rank="Average ranks",
        )
        print("Average ranks (lower is better):")
        print(avg_ranks)
    except Exception as e:
        print(f"Error drawing rank/CD diagrams: {e}")


    # Post-hoc tests (optional gate)
    if p_value < 0.05:
        print("\nSignificant differences found. Performing Wilcoxon and Nemenyi tests...")

        print("\nPerforming Wilcoxon signed-rank tests...")
        wilcoxon_results = perform_wilcoxon_tests(performance_df)
        print("Wilcoxon test results:")
        print(wilcoxon_results)

        print("\nPerforming Nemenyi post-hoc test...")
        nemenyi_results = perform_nemenyi_test(performance_df)
        print("Nemenyi test results:")
        print(nemenyi_results)

        wilcoxon_latex = create_latex_table(wilcoxon_results, "Wilcoxon Test Results", "tab:wilcoxon")
        nemenyi_latex = create_latex_table(nemenyi_results, "Nemenyi Test Results", "tab:nemenyi")

        print("\nWilcoxon Test Results (LaTeX):")
        print(wilcoxon_latex)
        print("\nNemenyi Test Results (LaTeX):")
        print(nemenyi_latex)
    else:
        print("\nNo significant differences found. Skipping Wilcoxon and Nemenyi tests.")


if __name__ == "__main__":
    main()