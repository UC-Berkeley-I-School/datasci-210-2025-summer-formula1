import json
import os
import pandas as pd
from pathlib import Path

def extract_target_class_metrics():
    # Base directory containing evaluation results
    base_dir = "/Users/seansica/Documents/Development/mids/capstone/datasci-210-2025-summer-formula1/sean/notebooks/week 9/evaluation_results"

    # List of experiment directories
    experiment_dirs = [
        # "20250708_133541_rocket_smote_driver16",
        # "20250708_133541_rocket_smote_driver4",
        # "20250708_133541_rocket_smote_driver55",
        # "20250708_133541_rocket_smote_driver63",
        # "20250708_133541_rocket_smote_driver81",
        # "20250708_133746_rocket_smote_driver23",
        # "20250708_133747_rocket_smote_driver1",
        # "20250708_133747_rocket_smote_driver10",
        # "20250708_133747_rocket_smote_driver22",
        # "20250708_133747_rocket_smote_driver44",
        # "20250708_133947_rocket_smote_driver2",
        # "20250708_133947_rocket_smote_driver3",
        # "20250708_133951_rocket_smote_driver14",
        # "20250708_133951_rocket_smote_driver18",
        # "20250708_133952_rocket_smote_driver77",
        # "20250708_134142_rocket_smote_driver20",
        # "20250708_134144_rocket_smote_driver11",
        # "20250708_134144_rocket_smote_driver24",
        # "20250708_134144_rocket_smote_driver27",
        # "20250708_134144_rocket_smote_driver31",
        # "20250708_154600_rocket_smote_driver16",
        # "20250708_154600_rocket_smote_driver4",
        # "20250708_154600_rocket_smote_driver55",
        # "20250708_154600_rocket_smote_driver63",
        # "20250708_154600_rocket_smote_driver81",
        # "20250708_154923_rocket_smote_driver1",
        # "20250708_154924_rocket_smote_driver10",
        # "20250708_154924_rocket_smote_driver22",
        # "20250708_154924_rocket_smote_driver23",
        # "20250708_154924_rocket_smote_driver44",
        # "20250708_155300_rocket_smote_driver2",
        # "20250708_155300_rocket_smote_driver3",
        # "20250708_155310_rocket_smote_driver14",
        # "20250708_155310_rocket_smote_driver18",
        # "20250708_155310_rocket_smote_driver77",
        # "20250708_155613_rocket_smote_driver20",
        # "20250708_155615_rocket_smote_driver11",
        # "20250708_155615_rocket_smote_driver24",
        # "20250708_155615_rocket_smote_driver27",
        # "20250708_155615_rocket_smote_driver31",
        # "20250708_161056_rocket_adasyn_driver16",
        # "20250708_161056_rocket_adasyn_driver4",
        # "20250708_161056_rocket_adasyn_driver55",
        # "20250708_161056_rocket_adasyn_driver63",
        # "20250708_161056_rocket_adasyn_driver81",
        # "20250708_161418_rocket_adasyn_driver1",
        # "20250708_161418_rocket_adasyn_driver10",
        # "20250708_161418_rocket_adasyn_driver44",
        # "20250708_161419_rocket_adasyn_driver22",
        # "20250708_161419_rocket_adasyn_driver23",
        # "20250708_161730_rocket_adasyn_driver2",
        # "20250708_161730_rocket_adasyn_driver3",
        # "20250708_161739_rocket_adasyn_driver14",
        # "20250708_161739_rocket_adasyn_driver77",
        # "20250708_161740_rocket_adasyn_driver18",
        # "20250708_162037_rocket_adasyn_driver20",
        # "20250708_162039_rocket_adasyn_driver11",
        # "20250708_162039_rocket_adasyn_driver24",
        # "20250708_162039_rocket_adasyn_driver27",
        # "20250708_162039_rocket_adasyn_driver31",
        "20250708_163306_catch22_rf_driver16",
        "20250708_163306_catch22_rf_driver55",
        "20250708_163306_catch22_rf_driver63",
        "20250708_163306_catch22_rf_driver81",
        "20250708_163307_catch22_rf_driver4",
        "20250708_163446_catch22_rf_driver1",
        "20250708_163447_catch22_rf_driver10",
        "20250708_163447_catch22_rf_driver22",
        "20250708_163447_catch22_rf_driver23",
        "20250708_163447_catch22_rf_driver44",
        "20250708_163618_catch22_rf_driver2",
        "20250708_163618_catch22_rf_driver3",
        "20250708_163623_catch22_rf_driver14",
        "20250708_163623_catch22_rf_driver18",
        "20250708_163623_catch22_rf_driver77",
        "20250708_163754_catch22_rf_driver20",
        "20250708_163755_catch22_rf_driver24",
        "20250708_163755_catch22_rf_driver27",
        "20250708_163755_catch22_rf_driver31",
        "20250708_163756_catch22_rf_driver11",
    ]

    results = []

    for exp_dir in experiment_dirs:
        exp_path = Path(base_dir) / exp_dir

        # Find the JSON files in the directory
        json_files = list(exp_path.glob("*.json"))

        complete_file = None
        external_file = None

        for json_file in json_files:
            if "external_complete.json" in json_file.name:
                external_file = json_file
            elif "complete.json" in json_file.name:
                complete_file = json_file

        # Extract driver number from directory name
        driver_num = exp_dir.split("_driver")[-1]

        row = {"experiment": exp_dir, "driver": driver_num}

        # Store file paths for hyperlinks
        if complete_file:
            row["test_file"] = complete_file.name
        if external_file:
            row["external_file"] = external_file.name

        # Extract test results (complete.json)
        if complete_file and complete_file.exists():
            try:
                with open(complete_file, 'r') as f:
                    data = json.load(f)

                    # target_class_metrics is inside the metrics key
                    metrics = data.get("metrics", {}).get("target_class_metrics", {})

                    row.update({
                        "test_tp": metrics.get("true_positives"),
                        "test_fn": metrics.get("false_negatives"),
                        "test_fp": metrics.get("false_positives"),
                        "test_tn": metrics.get("true_negatives"),
                        "test_precision": metrics.get("precision"),
                        "test_recall": metrics.get("recall"),
                        "test_f1": metrics.get("f1"),
                        "test_support": metrics.get("support")
                    })
            except Exception as e:
                print(f"Error reading {complete_file}: {e}")

        # Extract cross-eval results (external_complete.json)
        if external_file and external_file.exists():
            try:
                with open(external_file, 'r') as f:
                    data = json.load(f)
                    metrics = data.get("metrics", {}).get("target_class_metrics", {})

                    row.update({
                        "external_tp": metrics.get("true_positives"),
                        "external_fn": metrics.get("false_negatives"),
                        "external_fp": metrics.get("false_positives"),
                        "external_tn": metrics.get("true_negatives"),
                        "external_precision": metrics.get("precision"),
                        "external_recall": metrics.get("recall"),
                        "external_f1": metrics.get("f1"),
                        "external_support": metrics.get("support")
                    })
            except Exception as e:
                print(f"Error reading {external_file}: {e}")

        results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by driver number
    df = df.sort_values('driver', key=lambda x: x.astype(int))

    # Find top performers based on F1 score
    test_f1_max = df['test_f1'].max()
    external_f1_max = df['external_f1'].max()

    # Create formatted DataFrames with bold formatting for top performers
    def format_df_with_bold(df_subset, columns, f1_col, f1_max):
        formatted_df = df_subset.copy()

        for idx, row in formatted_df.iterrows():
            if row[f1_col] == f1_max:
                # Bold the entire row for the top performer
                for col in columns:
                    if col == 'driver':
                        formatted_df.at[idx, col] = f"**{row[col]}**"
                    elif pd.notna(row[col]):
                        if isinstance(row[col], (int, float)):
                            formatted_df.at[idx, col] = f"**{row[col]:.4f}**"
                        else:
                            formatted_df.at[idx, col] = f"**{row[col]}**"

        return formatted_df

    # Generate markdown table
    print("# Target Class Metrics Summary\n")
    print("## Test Results")
    test_cols = ['driver', 'test_precision', 'test_recall', 'test_f1', 'test_support']
    test_formatted = format_df_with_bold(df, test_cols, 'test_f1', test_f1_max)
    print(test_formatted[test_cols].to_markdown(index=False))

    print("\n## Cross-Evaluation Results")
    external_cols = ['driver', 'external_precision', 'external_recall', 'external_f1', 'external_support']
    external_formatted = format_df_with_bold(df, external_cols, 'external_f1', external_f1_max)
    print(external_formatted[external_cols].to_markdown(index=False))

    print("\n## Combined Results")
    combined_cols = ['driver', 'test_precision', 'test_recall', 'test_f1', 'external_precision', 'external_recall', 'external_f1']
    # For combined results, we need to handle both test and external bold formatting
    combined_formatted = df.copy()
    for idx, row in combined_formatted.iterrows():
        if row['test_f1'] == test_f1_max:
            combined_formatted.at[idx, 'driver'] = f"**{row['driver']}**"
            for col in ['test_precision', 'test_recall', 'test_f1']:
                if pd.notna(row[col]):
                    combined_formatted.at[idx, col] = f"**{row[col]:.4f}**"
        if row['external_f1'] == external_f1_max:
            combined_formatted.at[idx, 'driver'] = f"**{row['driver']}**"
            for col in ['external_precision', 'external_recall', 'external_f1']:
                if pd.notna(row[col]):
                    combined_formatted.at[idx, col] = f"**{row[col]:.4f}**"

    print(combined_formatted[combined_cols].to_markdown(index=False))

    # Save to file
    with open('target_class_metrics_summary.md', 'w') as f:
        f.write("# Target Class Metrics Summary\n\n")

        # Add source files section
        f.write("## Source Files\n\n")
        f.write("| Driver | Test Results File | Cross-Evaluation File |\n")
        f.write("|--------|-------------------|----------------------|\n")
        for _, row in df.iterrows():
            driver = row['driver']
            exp_dir = row['experiment']
            test_file = row.get('test_file', 'N/A')
            external_file = row.get('external_file', 'N/A')

            # Create hyperlinks to the files
            if test_file != 'N/A':
                test_link = f"[{test_file}](evaluation_results/{exp_dir}/{test_file})"
            else:
                test_link = "N/A"

            if external_file != 'N/A':
                external_link = f"[{external_file}](evaluation_results/{exp_dir}/{external_file})"
            else:
                external_link = "N/A"

            f.write(f"| {driver} | {test_link} | {external_link} |\n")

        f.write("\n## Test Results\n")
        f.write(test_formatted[test_cols].to_markdown(index=False))
        f.write("\n\n## Cross-Evaluation Results\n")
        f.write(external_formatted[external_cols].to_markdown(index=False))
        f.write("\n\n## Combined Results\n")
        f.write(combined_formatted[combined_cols].to_markdown(index=False))

    print("\nResults saved to target_class_metrics_summary.md")

    return df

if __name__ == "__main__":
    df = extract_target_class_metrics()
