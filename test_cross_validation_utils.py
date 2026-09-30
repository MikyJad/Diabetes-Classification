import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from cross_validation_utils import (
    CV_PROTOCOL, evaluate_cv, fold_plan, load_cv_result, metric_scores,
    nested_cv, save_cv_result, summary_row,
)


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        values, target = make_classification(n_samples=150, n_features=5, random_state=42)
        self.X = pd.DataFrame(values)
        self.y = pd.Series(target)
        self.cv = StratifiedKFold(5, shuffle=True, random_state=42)
        self.factory = lambda params: DecisionTreeClassifier(random_state=42, **params)
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    def evaluate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate_cv(self.factory, {'max_depth': 2}, self.X, self.y, self.cv)

    def test_each_row_is_tested_once_and_never_trained_on(self):
        class Memorizer:
            def fit(model, X, y):
                model.seen = set(X.index)
                return model

            def predict(model, X):
                self.assertFalse(model.seen.intersection(X.index))
                return np.zeros(len(X), dtype=int)

            def predict_proba(model, X):
                self.assertFalse(model.seen.intersection(X.index))
                return np.tile([0.6, 0.4], (len(X), 1))

        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_cv(lambda _: Memorizer(), {}, self.X, self.y, self.cv)
        np.testing.assert_array_equal(np.bincount(result['fold_id'])[1:], [30] * 5)

    def test_nested_objective_only_sees_outer_train(self):
        expected, _ = fold_plan(self.X, self.y, self.cv)
        seen = []

        def objective(trial, X_train, y_train, cv, groups_train):
            train, test = expected[len(seen)]
            self.assertEqual(set(X_train.index), set(self.X.index[train]))
            self.assertFalse(set(X_train.index).intersection(self.X.index[test]))
            seen.append(set(X_train.index))
            params = {'max_depth': trial.suggest_int('max_depth', 1, 2)}
            return cross_val_score(self.factory(params), X_train, y_train, cv=cv,
                                   scoring='f1').mean()

        with contextlib.redirect_stdout(io.StringIO()):
            result = nested_cv(objective, self.factory, self.X, self.y, self.cv,
                               self.cv, n_trials=1)
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(result['prediction']), len(self.y))

    def test_group_profiles_never_overlap_inner_or_outer(self):
        X = pd.concat([self.X, self.X], ignore_index=True)
        y = pd.concat([self.y, self.y], ignore_index=True)
        groups = np.tile(np.arange(len(self.X)), 2)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        outer, assignments = fold_plan(X, y, cv, groups)
        np.testing.assert_array_equal(assignments[:150], assignments[150:])
        for train, test in outer:
            self.assertFalse(set(groups[train]).intersection(groups[test]))
            inner, _ = fold_plan(X.iloc[train], y.iloc[train], cv, groups[train])
            for inner_train, inner_test in inner:
                self.assertFalse(set(groups[train][inner_train]).intersection(
                    groups[train][inner_test]))

    def test_summary_uses_fold_mean_and_sample_std(self):
        result = self.evaluate()
        summary = summary_row(result)
        for metric in metric_scores(self.y, result['prediction'], result['probability']):
            self.assertAlmostEqual(summary[metric], result['folds'][metric].mean())
            self.assertAlmostEqual(summary[f'{metric} Std'], result['folds'][metric].std(ddof=1))

    def test_ten_folds_nested_grouped_and_ungrouped(self):
        for grouped in [False, True]:
            with self.subTest(grouped=grouped):
                X = pd.concat([self.X, self.X], ignore_index=True) if grouped else self.X
                y = pd.concat([self.y, self.y], ignore_index=True) if grouped else self.y
                groups = np.tile(np.arange(len(self.X)), 2) if grouped else None
                splitter = StratifiedGroupKFold if grouped else StratifiedKFold
                cv = splitter(10, shuffle=True, random_state=42)
                seen = []

                def objective(trial, X_train, y_train, cv, groups_train):
                    self.assertEqual(cv.get_n_splits(), 10)
                    seen.append(len(X_train))
                    params = {'max_depth': trial.suggest_int('max_depth', 1, 2)}
                    return cross_val_score(
                        self.factory(params), X_train, y_train, cv=cv,
                        groups=groups_train, scoring='f1',
                    ).mean()

                with contextlib.redirect_stdout(io.StringIO()):
                    result = nested_cv(objective, self.factory, X, y, cv, cv,
                                       n_trials=1, groups=groups)
                self.assertEqual(len(seen), 10)
                self.assertEqual(result['inner_n_splits'], 10)
                self.assertEqual(summary_row(result)['Outer Folds'], 10)
                self.assertEqual(summary_row(result)['Inner Folds'], 10)
                if grouped:
                    np.testing.assert_array_equal(result['fold_id'][:150], result['fold_id'][150:])

    def test_export_roundtrip_and_stale_results_rejected(self):
        result = self.evaluate()
        with tempfile.TemporaryDirectory() as temporary:
            params = {'max_depth': 2}
            path = save_cv_result(result, self.X, self.y, None,
                                  Path(temporary) / 'result.json.gz', params)
            entry = {'params': params, 'evaluation_protocol': CV_PROTOCOL,
                     'cv_result_file': path}
            loaded = load_cv_result(entry, self.X, self.y, self.cv)
            np.testing.assert_array_equal(loaded['prediction'], result['prediction'])
            pd.testing.assert_frame_equal(loaded['folds'], result['folds'])
            changed_data = self.X.copy()
            changed_data.iloc[0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'data/parameter'):
                load_cv_result(entry, changed_data, self.y, self.cv)
            changed_entry = copy.deepcopy(entry)
            changed_entry['params']['max_depth'] = 3
            with self.assertRaisesRegex(ValueError, 'data/parameter'):
                load_cv_result(changed_entry, self.X, self.y, self.cv)
            with self.assertRaisesRegex(ValueError, 'protokol lama'):
                load_cv_result({'params': params}, self.X, self.y, self.cv)
            with self.assertRaisesRegex(ValueError, 'Jumlah fold'):
                load_cv_result(entry, self.X, self.y,
                               StratifiedKFold(10, shuffle=True, random_state=42))


if __name__ == '__main__':
    unittest.main()
