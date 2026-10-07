import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load telemetry datasets
df_local = pd.read_csv("local_benchmark_data.csv")
df_colab = pd.read_csv("colab_benchmark_data.csv")
df_jetstream = pd.read_csv("jetstream2_benchmark_data.csv")

# 2. Standardize column names to lowercase across all datasets
df_local.columns = df_local.columns.str.strip().str.lower()
df_colab.columns = df_colab.columns.str.strip().str.lower()
df_jetstream.columns = df_jetstream.columns.str.strip().str.lower()

# 3. Label the environments
df_local["environment"] = "Local (MacBook Pro)"
df_colab["environment"] = "Google Colab (T4 GPU)"
df_jetstream["environment"] = "Jetstream2 (A100 vGPU)"

# 4. Concatenate into a unified DataFrame
df = pd.concat([df_local, df_colab, df_jetstream], ignore_index=True)

# 5. Clean and sanitize throughput data
df["tokens_per_sec"] = pd.to_numeric(df["tokens_per_sec"], errors="coerce")
df_clean = df.dropna(subset=["tokens_per_sec"]).copy()

# Optional: Clean model names if any trailing tags exist
df_clean["model"] = df_clean["model"].astype(str)

# 6. Configure Plot Appearance
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(11, 6))

# Plot throughput across models and environments
ax = sns.barplot(
    data=df_clean,
    x="model",
    y="tokens_per_sec",
    hue="environment",
    palette="Blues_d",
    errorbar=None
)

# 7. Add Labels and Titles
plt.title("Comparative LLM Inference Throughput by Environment", fontsize=15, weight="bold", pad=15)
plt.xlabel("Model Evaluated", fontsize=12, labelpad=10)
plt.ylabel("Inference Speed (Tokens / Second)", fontsize=12, labelpad=10)
plt.legend(title="Execution Environment", loc="upper right", frameon=True)

# Add data labels on top of the bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.1f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points"
        )

plt.tight_layout()

# 8. Save Chart
chart_filename = "llm_throughput_comparison.png"
plt.savefig(chart_filename, dpi=300)
print(f"Chart successfully saved as: {chart_filename}")