import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import torch

from taborder import CausalOrderModel, SyntheticDataset, load_model, topological_divergence
from taborder import CausalOrderDAGModel


CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints/taborder-gp-small.pth"


def sample(seed=42, mode="additive-gp"):
    torch.manual_seed(seed)
    return next(iter(SyntheticDataset(
        variable_mode=mode, variable_range=(5, 5), parent_range=(1, 3),
        noise_var_range=(0.1, 0.5), batch_size=2,
        min_seq_len=32, max_seq_len=48, seed=seed,
    )))


class CoreTests(unittest.TestCase):
    def test_gp_generator_contract(self):
        for mode in ("additive-gp", "non-additive-gp"):
            batch = sample(mode=mode)
            x, rows, columns, truth, query, dags, masked, ancestors, valid, *_ = batch
            self.assertEqual(len(batch), 12)
            self.assertEqual(x.shape[-1], 4)
            self.assertTrue(x.isfinite().all())
            self.assertTrue(truth.isfinite().all())
            self.assertTrue(torch.equal(x[..., 1].bool(), masked))
            self.assertTrue(torch.equal(valid, (~rows).unsqueeze(-1).expand_as(valid)))
            torch.testing.assert_close(x[..., 0][valid & ~masked], truth[valid & ~masked])
            self.assertFalse((masked & rows.unsqueeze(-1)).any())
            self.assertEqual(columns.shape, (2, 5))
            self.assertTrue(query.any())
            for b, dag in enumerate(dags):
                self.assertTrue(nx.is_directed_acyclic_graph(dag))
                for node in dag:
                    self.assertEqual(set(torch.where(ancestors[b, node])[0].tolist()), nx.descendants(dag, node))

    def test_seeded_samples_reproduce(self):
        first, second = sample(), sample()
        for index in (0, 1, 2, 3, 4, 6, 7):
            self.assertTrue(torch.equal(first[index], second[index]))
        self.assertEqual(set(first[5][0].edges()), set(second[5][0].edges()))

    def test_checkpoint_and_predictions(self):
        model = load_model(CHECKPOINT)
        batch = sample()
        with torch.inference_mode():
            outputs = model(batch[0], row_mask=batch[1], col_mask=batch[2], query_indicator=batch[4])
        self.assertEqual(len(outputs), 7)
        self.assertTrue(all(output.isfinite().all() for output in outputs))
        self.assertEqual(outputs[2].shape, (2, 5))
        self.assertEqual(outputs[4].shape, (*batch[0].shape[:3], 2))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2899473)
        self.assertFalse(model.training)
        self.assertEqual(model.temperature_order, 0.1)

    def test_checkpoint_checksum(self):
        manifest = json.loads((CHECKPOINT.parent / "manifest.json").read_text())
        for entry in manifest["checkpoints"]:
            path = CHECKPOINT.parents[1] / entry["release_checkpoint"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["release_sha256"])

    def test_finetuned_checkpoint_and_predictions(self):
        model = load_model(CHECKPOINT.parent / "taborder-cc18-ctr23.pth")
        self.assertIsInstance(model, CausalOrderDAGModel)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2767888)
        batch = sample()
        with torch.inference_mode():
            outputs = model(batch[0], row_mask=batch[1], col_mask=batch[2], query_indicator=batch[4])
        self.assertEqual(len(outputs), 7)
        self.assertTrue(all(output.isfinite().all() for output in outputs))
        self.assertEqual(outputs[4].shape, (*batch[0].shape[:3], 2))

    def test_padding_does_not_change_valid_predictions(self):
        model = load_model(CHECKPOINT)
        batch = sample()
        x, rows, columns = batch[:3]
        with torch.inference_mode():
            full = model(x, row_mask=rows, col_mask=columns)
            for b in range(x.shape[0]):
                count = int((~rows[b]).sum())
                single = model(x[b:b+1, :count], row_mask=rows[b:b+1, :count], col_mask=columns[b:b+1])
                torch.testing.assert_close(full[2][b], single[2][0], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(full[4][b, :count], single[4][0], atol=2e-5, rtol=2e-5)

    def test_nll_backward_reaches_order_network(self):
        model = load_model(CHECKPOINT).train()
        batch = sample()
        outputs = model(batch[0], row_mask=batch[1], col_mask=batch[2], query_indicator=batch[4])
        predicted = outputs[4]
        mask = batch[6]
        loss = torch.nn.functional.gaussian_nll_loss(
            predicted[..., 0][mask], batch[3][mask], predicted[..., 1][mask].exp(),
        )
        loss.backward()
        grad = model.order_decoder[0].weight.grad
        self.assertTrue(grad.isfinite().all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_strict_load_rejects_incompatible_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pth"
            torch.save({"model_state_dict": {"unrelated": torch.zeros(1)}}, path)
            with self.assertRaises(RuntimeError):
                load_model(path)

    def test_topological_divergence_direction(self):
        dag = nx.DiGraph([(0, 1), (1, 2)])
        self.assertEqual(topological_divergence(torch.tensor([0., 1., 2.]), dag), 0)
        self.assertEqual(topological_divergence(torch.tensor([2., 1., 0.]), dag), 1)


if __name__ == "__main__":
    unittest.main()
