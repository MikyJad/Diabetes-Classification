import csv
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
MODEL_ORDER = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']
DATASET_ORDER = ['PIDD', 'FHGDD', 'BRFSS', 'Sylhet']


def as_source(text):
    return text.splitlines(keepends=True)


def load_notebook(path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def save_notebook(path, notebook):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
        handle.write('\n')
    os.replace(temporary_path, path)


def set_cell(notebook, index, source, cell_type='code'):
    cell = notebook['cells'][index]
    assert cell['cell_type'] == cell_type, (index, cell['cell_type'])
    cell['source'] = as_source(source)
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None


def replace_metric_import(source, include_curves=False):
    if include_curves:
        replacement = (
            'from sklearn.metrics import (\n'
            '    accuracy_score, precision_score, recall_score, f1_score,\n'
            '    roc_auc_score, average_precision_score, roc_curve,\n'
            '    precision_recall_curve\n'
            ')\n'
        )
    else:
        replacement = (
            'from sklearn.metrics import (accuracy_score, precision_score, recall_score,\n'
            '                             f1_score, roc_auc_score,\n'
            '                             average_precision_score)\n'
        )
    updated, count = re.subn(
        r'from sklearn\.metrics import \(.*?\)\r?\n',
        replacement,
        source,
        count=1,
        flags=re.DOTALL,
    )
    assert count == 1
    return updated


def update_model_notebooks():
    model_paths = sorted(
        path for path in ROOT.glob('*.ipynb')
        if not path.name.startswith('comparison_')
    )
    assert len(model_paths) == 16

    metric_helper = """METRIC_ORDER = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']

def calculate_metrics(y_true, prediction, probability):
    return {
        'Accuracy': accuracy_score(y_true, prediction),
        'Precision': precision_score(y_true, prediction, zero_division=0),
        'Recall': recall_score(y_true, prediction, zero_division=0),
        'F1-Score': f1_score(y_true, prediction, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, probability),
        'PR-AUC': average_precision_score(y_true, probability)
    }

def print_metrics(y_true, prediction, probability):
    metrics = calculate_metrics(y_true, prediction, probability)
    for metric in METRIC_ORDER:
        print(f'{metric:9s}: {metrics[metric]:.4f}')

print_metrics(y_test, y_pred, y_proba)"""

    baseline_plot = """baseline_metrics = calculate_metrics(y_test, y_pred, y_proba)
baseline_plot_df = pd.DataFrame({
    'Metric': METRIC_ORDER,
    'Score': [baseline_metrics[metric] for metric in METRIC_ORDER]
})

plt.figure(figsize=(10, 5))
ax = sns.barplot(data=baseline_plot_df, x='Metric', y='Score', color='steelblue')
ax.set_ylim(0, 1)
ax.set_title('Baseline Model Performance')
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=2)
plt.tight_layout()
plt.show()"""

    comparison_table = """performance_comparison = pd.DataFrame({
    'Baseline': calculate_metrics(y_test, y_pred, y_proba),
    'Tuned': calculate_metrics(y_test, optuna_prediction, optuna_proba)
}).T[METRIC_ORDER]
performance_comparison.round(4)"""

    comparison_plot = """performance_plot_df = (
    performance_comparison.reset_index(names='Model')
    .melt(id_vars='Model', var_name='Metric', value_name='Score')
)

plt.figure(figsize=(12, 5))
ax = sns.barplot(
    data=performance_plot_df, x='Metric', y='Score', hue='Model'
)
ax.set_ylim(0, 1)
ax.set_title('Baseline vs Tuned Model Performance')
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=2)
plt.tight_layout()
plt.show()"""

    final_table = """final_performance = pd.DataFrame(
    [calculate_metrics(y_test, optuna_prediction, optuna_proba)],
    index=['Tuned']
)[METRIC_ORDER]
final_performance.round(4)"""

    final_plot = """plt.figure(figsize=(10, 2.5))
sns.heatmap(
    final_performance, annot=True, fmt='.4f', cmap='YlGnBu',
    vmin=0, vmax=1, cbar_kws={'label': 'Score'}
)
plt.title('Final Tuned Model Performance')
plt.xlabel('Metric')
plt.ylabel('')
plt.tight_layout()
plt.show()"""

    for path in model_paths:
        notebook = load_notebook(path)
        original_cell_count = len(notebook['cells'])
        assert original_cell_count >= 25

        import_source = ''.join(notebook['cells'][0]['source'])
        set_cell(notebook, 0, replace_metric_import(import_source))
        set_cell(notebook, 16, metric_helper)
        set_cell(notebook, 17, baseline_plot)

        tuned_source = ''.join(notebook['cells'][20]['source'])
        tuned_lines = tuned_source.splitlines()
        assert len(tuned_lines) >= 2
        assert 'optuna_prediction =' in tuned_lines[0]
        assert 'optuna_proba =' in tuned_lines[1]
        tuned_display = '\n'.join(tuned_lines[:2]) + (
            '\n\nprint_metrics(y_test, optuna_prediction, optuna_proba)'
        )
        set_cell(notebook, 20, tuned_display)
        set_cell(notebook, 21, comparison_table)
        set_cell(notebook, 22, comparison_plot)
        set_cell(notebook, 23, final_table)
        set_cell(notebook, 24, final_plot)

        assert len(notebook['cells']) == original_cell_count
        save_notebook(path, notebook)


def comparison_eval_source(result_filename):
    return f"""y_test_array = y_test.to_numpy()
rows = []
for name in MODEL_ORDER:
    y_pred = predictions[name]
    y_proba = probabilities[name]
    rows.append({{
        'Model': name,
        'Accuracy': accuracy_score(y_test_array, y_pred),
        'Precision': precision_score(y_test_array, y_pred, zero_division=0),
        'Recall': recall_score(y_test_array, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test_array, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test_array, y_proba),
        'PR-AUC': average_precision_score(y_test_array, y_proba)
    }})

results_df = pd.DataFrame(rows, columns=['Model'] + METRIC_ORDER)
results_df.to_csv('{result_filename}', index=False)
results_df.round(4)"""


def probability_plot_source(dataset):
    return f"""metrics_to_plot = ['ROC-AUC', 'PR-AUC']
plot_df = results_df.melt(
    id_vars='Model', value_vars=metrics_to_plot,
    var_name='Metric', value_name='Score'
)

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=plot_df, x='Metric', y='Score', hue='Model', hue_order=MODEL_ORDER
)
ax.set_ylim(0, 1)
ax.set_title('Perbandingan ROC-AUC dan PR-AUC pada Test Set {dataset}')
ax.set_xlabel('Metrik')
ax.set_ylabel('Score')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()"""


def classification_plot_source(dataset):
    return f"""classification_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
classification_plot_df = results_df.melt(
    id_vars='Model', value_vars=classification_metrics,
    var_name='Metric', value_name='Score'
)

plt.figure(figsize=(11, 5))
ax = sns.barplot(
    data=classification_plot_df, x='Metric', y='Score', hue='Model',
    hue_order=MODEL_ORDER
)
ax.set_ylim(0, 1)
ax.set_title('Perbandingan Accuracy, Precision, Recall, dan F1-Score pada {dataset}')
ax.set_xlabel('Metrik')
ax.set_ylabel('Score')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()"""


def metric_heatmap_source(dataset):
    return f"""metric_matrix = (
    results_df.set_index('Model')[METRIC_ORDER]
    .reindex(MODEL_ORDER)
)

plt.figure(figsize=(10, 4))
sns.heatmap(
    metric_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
    vmin=0, vmax=1, cbar_kws={{'label': 'Score'}}
)
plt.title('Heatmap Enam Metrik Performa pada {dataset}')
plt.xlabel('Metrik')
plt.ylabel('Model')
plt.tight_layout()
plt.show()"""


def ranking_source():
    return """performance_ranking = (
    results_df.sort_values('F1-Score', ascending=False)
    .reset_index(drop=True)[['Model'] + METRIC_ORDER]
)
performance_ranking.round(4)"""


def update_roc_labels(source):
    source = source.replace('(AP=', '(PR-AUC=')
    source = source.replace("for name in models:", "for name in MODEL_ORDER:")
    return source


def update_dataset_comparison_notebooks():
    layouts = {
        'PIDD': {'roc': 13, 'heatmap': 15, 'ranking': 17, 'summary': 19,
                 'headings': {8: '## 4. Tabel enam metrik performa',
                              14: '## 6. Heatmap enam metrik performa',
                              16: '## 7. Urutan model berdasarkan F1-Score',
                              18: '## 8. Simpan prediksi dan ringkasan otomatis'}},
        'FHGDD': {'roc': 13, 'heatmap': 15, 'ranking': 17, 'summary': 19,
                  'headings': {8: '## 4. Tabel enam metrik performa',
                               14: '## 6. Heatmap enam metrik performa',
                               16: '## 7. Urutan model berdasarkan F1-Score',
                               18: '## 8. Simpan prediksi dan ringkasan otomatis'}},
        'BRFSS': {'roc': 13, 'heatmap': 15, 'ranking': 17, 'summary': 19,
                  'headings': {8: '## 4. Tabel enam metrik performa',
                               14: '## 6. Heatmap enam metrik performa',
                               16: '## 7. Urutan model berdasarkan F1-Score',
                               18: '## 8. Simpan prediksi dan ringkasan otomatis'}},
        'Sylhet': {'roc': 12, 'heatmap': 13, 'ranking': 15, 'summary': 17,
                   'headings': {8: '## Enam metrik performa pada test set',
                                14: '## Urutan model berdasarkan F1-Score',
                                16: '## Ekspor hasil dan ringkasan'}},
    }

    for dataset, layout in layouts.items():
        path = ROOT / f'comparison_{dataset}.ipynb'
        notebook = load_notebook(path)
        original_cell_count = len(notebook['cells'])

        import_source = ''.join(notebook['cells'][1]['source'])
        import_source = import_source.replace('from itertools import combinations\n\n', '')
        import_source = import_source.replace('from scipy.stats import binomtest\n\n', '')
        import_source = replace_metric_import(import_source, include_curves=True)
        if 'METRIC_ORDER =' not in import_source:
            import_source = import_source.replace(
                "MODEL_ORDER = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']",
                "MODEL_ORDER = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']\n"
                "METRIC_ORDER = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']",
            )
        set_cell(notebook, 1, import_source)
        set_cell(
            notebook, 9,
            comparison_eval_source(f'comparison_{dataset}_results.csv')
        )
        set_cell(notebook, 10, probability_plot_source(dataset))
        set_cell(notebook, 11, classification_plot_source(dataset))

        roc_source = ''.join(notebook['cells'][layout['roc']]['source'])
        set_cell(notebook, layout['roc'], update_roc_labels(roc_source))
        set_cell(notebook, layout['heatmap'], metric_heatmap_source(dataset))
        set_cell(notebook, layout['ranking'], ranking_source())

        summary_index = layout['summary']
        summary_source = ''.join(notebook['cells'][summary_index]['source'])
        prediction_prefix = summary_source.split("best_test_f1 =", 1)[0]
        assert "prediction_df.to_csv" in prediction_prefix
        summary = prediction_prefix.rstrip() + """

for metric in METRIC_ORDER:
    best_model = results_df.loc[results_df[metric].idxmax(), 'Model']
    print(f'Model dengan {metric:9s} tertinggi: {best_model}')
print('Hasil tersimpan di comparison_%s_results.csv')
print('Prediksi tersimpan di comparison_%s_predictions.csv')""" % (dataset, dataset)
        set_cell(notebook, summary_index, summary)

        for index, heading in layout['headings'].items():
            set_cell(notebook, index, heading, cell_type='markdown')

        assert len(notebook['cells']) == original_cell_count
        save_notebook(path, notebook)


def update_cross_dataset_notebook():
    path = ROOT / 'comparison_datasets_per_model.ipynb'
    notebook = load_notebook(path)
    original_cell_count = len(notebook['cells'])

    import_source = ''.join(notebook['cells'][1]['source'])
    if 'METRIC_ORDER =' not in import_source:
        import_source = import_source.replace(
            "MODEL_ORDER = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']",
            "MODEL_ORDER = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']\n"
            "METRIC_ORDER = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']",
        )
    set_cell(notebook, 1, import_source)

    cell3 = ''.join(notebook['cells'][3]['source'])
    prefix = cell3.split("all_results['Positive Prevalence']", 1)[0]
    assert "prevalence_map" in prefix
    prefix = prefix.split('prevalence_map =', 1)[0].rstrip()
    cell3_updated = prefix + """

all_results = all_results[['Dataset', 'Model'] + METRIC_ORDER]
all_results.to_csv('comparison_datasets_per_model_results.csv', index=False)

assert len(all_results) == 16
assert all_results.groupby('Dataset')['Model'].nunique().eq(4).all()
profile_df"""
    set_cell(notebook, 3, cell3_updated)

    set_cell(notebook, 5, """comparison_columns = ['Dataset', 'Model'] + METRIC_ORDER
all_results[comparison_columns].sort_values(['Model', 'Dataset']).round(4)""")

    set_cell(notebook, 6, """per_model_columns = ['Dataset'] + METRIC_ORDER
for model in MODEL_ORDER:
    display(Markdown(f'### {model}'))
    model_table = (
        all_results.loc[all_results['Model'].eq(model), per_model_columns]
        .set_index('Dataset')
        .reindex(DATASET_ORDER)
    )
    display(model_table.round(4))""")

    set_cell(notebook, 8, """classification_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
for ax, model in zip(axes.ravel(), MODEL_ORDER):
    model_plot = all_results.loc[
        all_results['Model'].eq(model), ['Dataset'] + classification_metrics
    ].melt(id_vars='Dataset', var_name='Metric', value_name='Score')
    sns.barplot(
        data=model_plot, x='Metric', y='Score', hue='Dataset',
        hue_order=DATASET_ORDER, ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title(model)
    ax.set_xlabel('Metrik')
    ax.set_ylabel('Score')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=7, padding=1)
    ax.legend(title='Dataset', loc='lower right')
fig.suptitle(
    'Perbandingan Dataset: Accuracy, Precision, Recall, dan F1-Score',
    fontsize=14, y=1.01
)
plt.tight_layout()
plt.show()""")

    set_cell(notebook, 9, """fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, metric in zip(axes.ravel(), METRIC_ORDER):
    matrix = (
        all_results.pivot(index='Dataset', columns='Model', values=metric)
        .reindex(index=DATASET_ORDER, columns=MODEL_ORDER)
    )
    sns.heatmap(
        matrix, annot=True, fmt='.3f', cmap='YlGnBu',
        vmin=0, vmax=1, ax=ax
    )
    ax.set_title(metric)
    ax.set_xlabel('Model')
    ax.set_ylabel('Dataset')
plt.tight_layout()
plt.show()""")

    set_cell(notebook, 11, """dataset_summary = (
    all_results.groupby('Dataset', as_index=False, observed=False)
    .agg(
        Mean_Accuracy=('Accuracy', 'mean'),
        Mean_Precision=('Precision', 'mean'),
        Mean_Recall=('Recall', 'mean'),
        Mean_F1_Score=('F1-Score', 'mean'),
        Mean_ROC_AUC=('ROC-AUC', 'mean'),
        Mean_PR_AUC=('PR-AUC', 'mean')
    )
)
dataset_summary['Dataset'] = pd.Categorical(
    dataset_summary['Dataset'], categories=DATASET_ORDER, ordered=True
)
dataset_summary = dataset_summary.sort_values('Dataset').reset_index(drop=True)
dataset_summary.to_csv('comparison_dataset_summary.csv', index=False)
dataset_summary.round(4)""")

    set_cell(notebook, 12, """summary_columns = {
    'Accuracy': 'Mean_Accuracy',
    'Precision': 'Mean_Precision',
    'Recall': 'Mean_Recall',
    'F1-Score': 'Mean_F1_Score',
    'ROC-AUC': 'Mean_ROC_AUC',
    'PR-AUC': 'Mean_PR_AUC'
}

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, (metric, column) in zip(axes.ravel(), summary_columns.items()):
    sns.barplot(
        data=dataset_summary, x='Dataset', y=column,
        order=DATASET_ORDER, ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title(f'Rata-rata {metric}')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=2)
plt.tight_layout()
plt.show()""")

    set_cell(notebook, 14, """decision_rows = []
for model in MODEL_ORDER:
    model_results = all_results.loc[all_results['Model'].eq(model)]
    row = {'Model': model}
    for metric in METRIC_ORDER:
        row[f'{metric} tertinggi'] = model_results.loc[
            model_results[metric].idxmax(), 'Dataset'
        ]
    decision_rows.append(row)

decision_df = pd.DataFrame(decision_rows)
display(decision_df)

summary_columns = {
    'Accuracy': 'Mean_Accuracy',
    'Precision': 'Mean_Precision',
    'Recall': 'Mean_Recall',
    'F1-Score': 'Mean_F1_Score',
    'ROC-AUC': 'Mean_ROC_AUC',
    'PR-AUC': 'Mean_PR_AUC'
}
for metric, column in summary_columns.items():
    winner = dataset_summary.loc[dataset_summary[column].idxmax(), 'Dataset']
    print(f'{metric:9s} rata-rata tertinggi: {winner}')""")

    set_cell(notebook, 10, '## 4. Ringkasan rata-rata enam metrik', cell_type='markdown')
    set_cell(notebook, 13, '## 5. Dataset terbaik untuk setiap metrik', cell_type='markdown')

    assert len(notebook['cells']) == original_cell_count
    save_notebook(path, notebook)


def read_csv_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, fieldnames, rows):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def metric_value(row, metric):
    if metric == 'F1-Score':
        return row.get('F1-Score') or row['F1']
    return row[metric]


def update_result_csvs():
    combined_rows = []
    for dataset in DATASET_ORDER:
        path = ROOT / f'comparison_{dataset}_results.csv'
        existing_rows = read_csv_rows(path)
        by_model = {row['Model']: row for row in existing_rows}
        assert set(by_model) == set(MODEL_ORDER)
        cleaned_rows = []
        for model in MODEL_ORDER:
            old_row = by_model[model]
            new_row = {'Model': model}
            new_row.update({metric: metric_value(old_row, metric) for metric in METRICS})
            cleaned_rows.append(new_row)
            combined_rows.append({'Dataset': dataset, **new_row})
        write_csv_rows(path, ['Model'] + METRICS, cleaned_rows)

    write_csv_rows(
        ROOT / 'comparison_datasets_per_model_results.csv',
        ['Dataset', 'Model'] + METRICS,
        combined_rows,
    )

    summary_rows = []
    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in combined_rows if row['Dataset'] == dataset]
        summary = {'Dataset': dataset}
        summary_names = {
            'Accuracy': 'Mean_Accuracy',
            'Precision': 'Mean_Precision',
            'Recall': 'Mean_Recall',
            'F1-Score': 'Mean_F1_Score',
            'ROC-AUC': 'Mean_ROC_AUC',
            'PR-AUC': 'Mean_PR_AUC',
        }
        for metric, summary_name in summary_names.items():
            summary[summary_name] = sum(
                float(row[metric]) for row in dataset_rows
            ) / len(dataset_rows)
        summary_rows.append(summary)

    write_csv_rows(
        ROOT / 'comparison_dataset_summary.csv',
        [
            'Dataset', 'Mean_Accuracy', 'Mean_Precision', 'Mean_Recall',
            'Mean_F1_Score', 'Mean_ROC_AUC', 'Mean_PR_AUC'
        ],
        summary_rows,
    )


def main():
    update_model_notebooks()
    update_dataset_comparison_notebooks()
    update_result_csvs()
    update_cross_dataset_notebook()


if __name__ == '__main__':
    main()
