import unittest
from pathlib import Path

import numpy as np
import torch

from taborder import encode_table, impute_table, load_model


CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints/taborder-gp-small.pth"


class InferenceTests(unittest.TestCase):
    def test_hidden_truth_and_query_rows_do_not_fit_preprocessing(self):
        values = np.arange(72, dtype=float).reshape(24, 3)
        missing = np.zeros(values.shape, dtype=bool)
        missing[1, 0] = True
        missing[12:, 1] = True
        first = encode_table(values, missing, context_size=12)
        altered = values.copy()
        altered[missing] = np.nan
        second = encode_table(altered, missing, context_size=12)
        torch.testing.assert_close(first.x, second.x, atol=0, rtol=0)
        altered[12:, 0] += 100000
        third = encode_table(altered, missing, context_size=12)
        np.testing.assert_array_equal(first.mean, third.mean)
        np.testing.assert_array_equal(first.scale, third.scale)

    def test_unobserved_context_column_is_rejected(self):
        values = np.ones((16, 3))
        missing = np.zeros_like(values, dtype=bool)
        missing[:8, 2] = True
        with self.assertRaisesRegex(ValueError, "no observed context"):
            encode_table(values, missing, context_size=8)

    def test_observed_values_preserved(self):
        model = load_model(CHECKPOINT)
        values = np.random.default_rng(5).normal(size=(32, 3))
        missing = np.zeros(values.shape, dtype=bool)
        missing[16:, 1] = True
        result = impute_table(model, np.where(missing, np.nan, values), missing, context_size=16)
        self.assertTrue(np.isfinite(result["imputed"]).all())
        np.testing.assert_array_equal(result["imputed"][~missing], values[~missing])

    def test_fixed_causal_mask_blocks_descendant_effect_on_mean(self):
        model = load_model(CHECKPOINT)
        values = np.random.default_rng(8).normal(size=(32, 3))
        missing = np.zeros(values.shape, dtype=bool)
        missing[16:, 1] = True
        order = np.array([[0., 1., 2.]])
        first = impute_table(model, values, missing, context_size=16, order_scores_override=order)
        values[16:, 2] += 1000
        second = impute_table(model, values, missing, context_size=16, order_scores_override=order)
        np.testing.assert_allclose(first["predicted_mean"][16:, 1], second["predicted_mean"][16:, 1], atol=1e-5, rtol=1e-5)

    def test_default_forward_unchanged_by_reusing_its_order_scores(self):
        model = load_model(CHECKPOINT)
        values = np.random.default_rng(2).normal(size=(16, 3))
        table = encode_table(values, np.zeros_like(values, dtype=bool), context_size=16)
        with torch.inference_mode():
            first = model(table.x, row_mask=table.row_mask, col_mask=table.col_mask)
            second = model(table.x, row_mask=table.row_mask, col_mask=table.col_mask,
                           order_scores_override=first[2])
        for original, repeated in zip(first, second):
            torch.testing.assert_close(original, repeated, atol=0, rtol=0)

    def test_numpy_order_override_supported_by_both_variants(self):
        values = np.random.default_rng(3).normal(size=(16, 3))
        missing = np.zeros(values.shape, dtype=bool)
        missing[8:, 1] = True
        for filename in ("taborder-gp-small.pth", "taborder-cc18-ctr23.pth"):
            model = load_model(CHECKPOINT.parent / filename)
            result = impute_table(model, values, missing, context_size=8,
                                  order_scores_override=np.array([[0., 1., 2.]]))
            self.assertTrue(np.isfinite(result["imputed"]).all())


if __name__ == "__main__":
    unittest.main()
