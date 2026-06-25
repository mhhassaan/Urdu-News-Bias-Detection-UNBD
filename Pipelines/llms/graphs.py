import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# =========================================================
# GLOBAL STYLE
# =========================================================

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10
})

# =========================================================
# LOAD SONNET CSV FILES
# =========================================================

zero_df = pd.read_csv("sonnet_zero_shot.csv")
few_df = pd.read_csv("sonnet_few_shot.csv")

# =========================================================
# STANDARDIZE LABELS
# =========================================================

for df in [zero_df, few_df]:

    df['Label'] = (
        df['Label']
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df['llm'] = (
        df['llm']
        .astype(str)
        .str.lower()
        .str.strip()
    )

# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(df):

    y_true = df['Label']
    y_pred = df['llm']

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average='weighted'
        )
    )

    return {
        "Accuracy": round(accuracy * 100, 2),
        "F1-Score": round(f1 * 100, 2)
    }

# =========================================================
# SONNET METRICS
# =========================================================

sonnet_zero = calculate_metrics(zero_df)
sonnet_few = calculate_metrics(few_df)

# =========================================================
# CREATE DATAFRAMES
# =========================================================

zero_shot_results = pd.DataFrame({

    "Model": [
        "Grok",
        "Gemini-3-Thinking",
        "Claude Sonnet 4.6"
    ],

    "Accuracy": [
        57.92,
        57.31,
        sonnet_zero["Accuracy"]
    ],

    "Weighted F1": [
        57.33,
        49.00,
        sonnet_zero["F1-Score"]
    ]
})

few_shot_results = pd.DataFrame({

    "Model": [
        "Grok",
        "Gemini-3-Thinking",
        "Claude Sonnet 4.6"
    ],

    "Accuracy": [
        57.56,
        55.11,
        sonnet_few["Accuracy"]
    ],

    "Weighted F1": [
        56.62,
        48.00,
        sonnet_few["F1-Score"]
    ]
})

# =========================================================
# PROFESSIONAL GRAPH FUNCTION
# =========================================================

def create_research_graph(df, title, save_path):

    # -----------------------------------------------------
    # Melt Data
    # -----------------------------------------------------

    melted = df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    # -----------------------------------------------------
    # Create Figure
    # -----------------------------------------------------

    plt.figure(figsize=(9, 6))

    ax = sns.barplot(
        data=melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette=["#4C72B0", "#DD8452"]
    )

    # -----------------------------------------------------
    # Add Labels on Bars
    # -----------------------------------------------------

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt='%.1f',
            padding=3,
            fontsize=9
        )

    # -----------------------------------------------------
    # Styling
    # -----------------------------------------------------

    plt.title(
        title,
        pad=14,
        weight='bold'
    )

    plt.ylabel("Performance Score (%)")
    plt.xlabel("")

    plt.ylim(40, 100)

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.35
    )

    # -----------------------------------------------------
    # Clean Borders
    # -----------------------------------------------------

    sns.despine()

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------

    plt.legend(
        title="Evaluation Metric",
        frameon=True
    )

    # -----------------------------------------------------
    # Tight Layout
    # -----------------------------------------------------

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()

# =========================================================
# GENERATE PROFESSIONAL FIGURES
# =========================================================

create_research_graph(
    zero_shot_results,
    "Comparative Performance of Large Language Models",
    "llm_comparison_graph_1.png"
)

create_research_graph(
    few_shot_results,
    "Performance Evaluation of Large Language Models",
    "llm_comparison_graph_2.png"
)

print("Professional research-style graphs generated.")