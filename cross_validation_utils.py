"""Evaluasi out-of-fold bersama untuk notebook model dan komparasi."""

import copy
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


CV_PROTOCOL = 'nested-oof-v2'
METRICS = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall',
           'Specificity', 'F1', 'ROC-AUC', 'PR-AUC']


def fold_plan(X, y, cv, groups=None):
    """Pastikan setiap baris diuji sekali dan tidak ada group leakage."""
    splits = list(cv.split(X, y, groups=groups))
    coverage = np.zeros(len(y), dtype=int)
    assignments = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(splits, start=1):
        if len(np.intersect1d(train, test)):
            raise ValueError('Indeks train dan validation bertumpang tindih.')
        if len(np.unique(np.asarray(y)[test])) != 2:
            raise ValueError('Setiap validation fold harus memiliki dua kelas.')
        if groups is not None:
            group_values = np.asarray(groups)
            if len(np.intersect1d(group_values[train], group_values[test])):
                raise ValueError('Profil identik bocor antar-fold.')
        coverage[test] += 1
        assignments[test] = fold
    if not np.all(coverage == 1):
        raise ValueError('Setiap baris harus menjadi data uji tepat sekali.')
    return splits, assignments


def split_summary(X, y, cv, groups=None):
    splits, _ = fold_plan(X, y, cv, groups)
    return pd.DataFrame([
        {'Fold': fold, 'Train': len(train), 'Validation': len(test),
         'Positif train': int(np.asarray(y)[train].sum()),
         'Positif validation': int(np.asarray(y)[test].sum())}
        for fold, (train, test) in enumerate(splits, start=1)
    ])


def metric_scores(y, prediction, probability):
    tn, fp, _, _ = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y, prediction),
        'Balanced Accuracy': balanced_accuracy_score(y, prediction),
        'Precision': precision_score(y, prediction, zero_division=0),
        'Recall': recall_score(y, prediction, zero_division=0),
        'Specificity': tn / (tn + fp),
        'F1': f1_score(y, prediction, zero_division=0),
        'ROC-AUC': roc_auc_score(y, probability),
        'PR-AUC': average_precision_score(y, probability),
    }


def _evaluate(make_pipeline, X, y, cv, groups, tune):
    splits, assignments = fold_plan(X, y, cv, groups)
    prediction = np.empty(len(y), dtype=int)
    probability = np.empty(len(y), dtype=float)
    rows = []
    for fold, (train, test) in enumerate(splits, start=1):
        started = time.perf_counter()
        print(f'Outer fold {fold}/{len(splits)}: train={len(train)}, '
              f'validation={len(test)}', flush=True)
        X_train, y_train = X.iloc[train], y.iloc[train]
        groups_train = None if groups is None else np.asarray(groups)[train]
        params, inner_f1 = tune(X_train, y_train, groups_train)
        pipeline = make_pipeline(params)
        # Imputer, scaler, dan SMOTE dipelajari hanya di outer training fold.
        pipeline.fit(X_train, y_train)
        fold_prediction = np.asarray(pipeline.predict(X.iloc[test])).astype(int).ravel()
        fold_probability = pipeline.predict_proba(X.iloc[test])[:, 1]
        prediction[test] = fold_prediction
        probability[test] = fold_probability
        rows.append({
            'Fold': fold, 'Train Rows': len(train), 'Validation Rows': len(test),
            **metric_scores(y.iloc[test], fold_prediction, fold_probability),
            'Fit seconds': time.perf_counter() - started,
            'Inner CV F1': inner_f1, 'Best Params': params,
        })
    return {'prediction': prediction, 'probability': probability,
            'fold_id': assignments, 'folds': pd.DataFrame(rows)}


def evaluate_cv(make_pipeline, params, X, y, cv, groups=None):
    return _evaluate(make_pipeline, X, y, cv, groups,
                     lambda *_: (dict(params), None))


def nested_cv(objective, make_pipeline, X, y, outer_cv, inner_cv,
              n_trials, groups=None, random_state=42):
    import optuna

    def tune(X_train, y_train, groups_train):
        cv = copy.deepcopy(inner_cv)
        # Periksa juga coverage dan group overlap pada inner folds.
        fold_plan(X_train, y_train, cv, groups_train)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, cv, groups_train),
            n_trials=n_trials,
        )
        return dict(study.best_params), float(study.best_value)

    result = _evaluate(make_pipeline, X, y, outer_cv, groups, tune)
    result['inner_n_splits'] = inner_cv.get_n_splits()
    return result


def summary_row(result):
    """Metrik utama adalah mean fold, bukan metrik pooled OOF."""
    folds = result['folds']
    row = {}
    for metric in METRICS:
        row[metric] = float(folds[metric].mean())
        row[f'{metric} Std'] = float(folds[metric].std(ddof=1))
    row['CV F1'] = row['F1']
    row['Fit seconds'] = float(folds['Fit seconds'].sum())
    row['Outer Folds'] = len(folds)
    row['Inner Folds'] = result.get('inner_n_splits')
    row['Evaluation Protocol'] = CV_PROTOCOL
    return row


def print_cv_metrics(result):
    row = summary_row(result)
    print('Cross-Validation Results (Mean +/- Std):')
    for metric in METRICS:
        print(f'{metric:18}: {row[metric]:.4f} +/- {row[f"{metric} Std"]:.4f}')


def _fingerprint(X, y, groups):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(X.columns)).encode('utf-8'))
    digest.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    digest.update(np.asarray(y, dtype=np.int64).tobytes())
    if groups is not None:
        digest.update(np.asarray(groups, dtype=np.int64).tobytes())
    return digest.hexdigest()


def save_cv_result(result, X, y, groups, path, best_params):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'protocol': CV_PROTOCOL, 'fingerprint': _fingerprint(X, y, groups),
        'best_params': best_params,
        'outer_n_splits': len(result['folds']),
        'inner_n_splits': result.get('inner_n_splits'),
        'prediction': result['prediction'].tolist(),
        'probability': result['probability'].tolist(),
        'fold_id': result['fold_id'].tolist(),
        'folds': result['folds'].to_dict(orient='records'),
    }
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle)
    return path.as_posix()


def load_cv_result(entry, X, y, cv, groups=None):
    n_splits = cv.get_n_splits()
    message = ('Jalankan ulang notebook model dari atas sampai cell penyimpanan '
               f'hasil tuning untuk menghasilkan evaluasi nested {n_splits}-fold terbaru.')
    if entry.get('evaluation_protocol') != CV_PROTOCOL:
        raise ValueError(message + ' JSON yang tersedia masih memakai protokol lama.')
    path = Path(entry['cv_result_file'])
    if not path.exists():
        raise FileNotFoundError(f'{path} belum tersedia. {message}')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        payload = json.load(handle)
    if (payload['protocol'] != CV_PROTOCOL or
            payload['fingerprint'] != _fingerprint(X, y, groups) or
            payload['best_params'] != entry['params']):
        raise ValueError('Hasil CV tidak cocok dengan data/parameter terbaru. ' + message)
    if (payload['outer_n_splits'] != n_splits or
            payload['inner_n_splits'] not in [None, n_splits]):
        raise ValueError('Jumlah fold berbeda dengan konfigurasi terbaru. ' + message)
    result = {
        'prediction': np.asarray(payload['prediction'], dtype=int),
        'probability': np.asarray(payload['probability'], dtype=float),
        'fold_id': np.asarray(payload['fold_id'], dtype=int),
        'folds': pd.DataFrame(payload['folds']),
        'inner_n_splits': payload['inner_n_splits'],
    }
    splits, assignments = fold_plan(X, y, cv, groups)
    if not np.array_equal(result['fold_id'], assignments):
        raise ValueError('Pembagian fold berbeda dengan notebook model. ' + message)
    if any(len(result[key]) != len(y) for key in ['prediction', 'probability']):
        raise ValueError('Jumlah prediksi OOF tidak lengkap. ' + message)
    if (not np.isfinite(result['probability']).all() or
            not np.isin(result['prediction'], [0, 1]).all() or
            np.any((result['probability'] < 0) | (result['probability'] > 1))):
        raise ValueError('Prediksi/probabilitas OOF tidak valid.')
    if list(result['folds']['Fold']) != list(range(1, len(splits) + 1)):
        raise ValueError('Tabel metrik fold tidak lengkap. ' + message)
    for fold, (_, test) in enumerate(splits):
        expected = metric_scores(y.iloc[test], result['prediction'][test],
                                 result['probability'][test])
        for metric, value in expected.items():
            if not np.isclose(result['folds'].iloc[fold][metric], value):
                raise ValueError('Metrik fold tidak cocok dengan prediksi OOF.')
    return result
