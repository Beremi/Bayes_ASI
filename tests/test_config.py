from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bayes_asi.config import HMCConfig, QueueConfig, SurrogateConfig, load_config, resolve_dotted


MINIMAL_CONFIG = """
problem:
  latent_dim: 1
  parameter_dim: 1
  output_dim: 1
solver:
  factory: bayes_asi.examples.synthetic:make_solver
posterior:
  factory: bayes_asi.examples.synthetic:make_posterior
"""


class ConfigTests(unittest.TestCase):
    def _write(self, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_minimal_config_uses_valid_defaults(self) -> None:
        config = load_config(self._write(MINIMAL_CONFIG))
        self.assertEqual(config.problem.latent_dim, 1)
        self.assertEqual(config.surrogate.hidden_layers, [64, 64])
        self.assertTrue(config.run.output_dir.is_absolute())

    def test_unknown_option_is_rejected(self) -> None:
        path = self._write(MINIMAL_CONFIG + "\nsurrogate:\n  learnng_rate: 0.1\n")
        with self.assertRaisesRegex(ValueError, "learnng_rate"):
            load_config(path)

    def test_non_mapping_section_is_rejected(self) -> None:
        path = self._write(MINIMAL_CONFIG + "\nqueue: invalid\n")
        with self.assertRaisesRegex(ValueError, "queue must be a mapping"):
            load_config(path)

    def test_quoted_boolean_is_rejected(self) -> None:
        path = self._write(MINIMAL_CONFIG + '\nrun:\n  drain_running_on_exit: "false"\n')
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            load_config(path)

    def test_invalid_numeric_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            SurrogateConfig(validation_fraction=1.0)
        with self.assertRaisesRegex(ValueError, "failure_retry_seconds"):
            SurrogateConfig(failure_retry_seconds=-1.0)
        with self.assertRaisesRegex(ValueError, "leapfrog"):
            HMCConfig(min_leapfrog_steps=5, max_leapfrog_steps=4)
        with self.assertRaisesRegex(ValueError, "max_pending"):
            QueueConfig(min_pending_per_worker=2, max_pending_per_worker=1)

    def test_factory_path_validation_and_resolution(self) -> None:
        self.assertIs(resolve_dotted("bayes_asi.config:load_config"), load_config)
        with self.assertRaisesRegex(ValueError, "module and callable"):
            resolve_dotted("invalid")


if __name__ == "__main__":
    unittest.main()
