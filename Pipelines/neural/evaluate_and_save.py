def evaluate_and_save(
    pipeline_name,
    grid,
    best_model,
    X,
    y,
    y_pred,
    kfold,
    num_features=None,
    results_dir="results"
):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    os.makedirs(results_dir, exist_ok=True)

    print(f"\n📊 FINAL RESULTS: {pipeline_name}")
    print("="*60)

    # ===============================
    # METRICS
    # ===============================
    accuracy = accuracy_score(y, y_pred)

    # ✅ FIX: use predictions directly (NO cross_val_score)
    cv_accuracy = accuracy  

    best_f1 = grid.best_score_

    if num_features is not None:
        print(f"Total Features: {num_features}")

    # ===============================
    # REPORT
    # ===============================
    report_text = classification_report(
        y, y_pred,
        target_names=['Unbiased', 'Biased']
    )

    report_dict = classification_report(
        y, y_pred,
        target_names=['Unbiased', 'Biased'],
        output_dict=True
    )

    print("\nBest Parameters:")
    print(grid.best_params_)

    print(f"\nBest CV F1 (macro): {best_f1:.4f}")
    print(f"CV Accuracy: {cv_accuracy:.4f}")
    print(f"Final Accuracy (5-Fold CV): {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(report_text)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unbiased','Biased'],
                yticklabels=['Unbiased','Biased'])

    plt.title(f'{pipeline_name} (Acc: {accuracy:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig(f"{results_dir}/{pipeline_name}_confusion_matrix.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # ===============================
    # SAVE CSV FILES
    # ===============================
    pd.DataFrame(report_dict).transpose().to_csv(
        f"{results_dir}/{pipeline_name}_classification_report.csv"
    )

    pd.DataFrame(cm,
                 index=['Unbiased','Biased'],
                 columns=['Unbiased','Biased']).to_csv(
        f"{results_dir}/{pipeline_name}_confusion_matrix.csv"
    )

    # ===============================
    # SAVE TXT LOG
    # ===============================
    with open(f"{results_dir}/{pipeline_name}_full_report.txt", "w") as f:

        f.write(f"===== {pipeline_name} =====\n\n")

        if num_features is not None:
            f.write(f"Total Features: {num_features}\n\n")

        f.write("Parameter Grid:\n")
        f.write(str(grid.param_grid) + "\n\n")

        f.write("Best Parameters:\n")
        for k, v in grid.best_params_.items():
            f.write(f"{k}: {v}\n")

        f.write(f"\nBest CV F1 (macro): {best_f1:.4f}\n")
        f.write(f"CV Accuracy: {cv_accuracy:.4f}\n")
        f.write(f"Final Accuracy (5-Fold CV): {accuracy:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report_text)

    print(f"\n✅ Results saved for {pipeline_name}")