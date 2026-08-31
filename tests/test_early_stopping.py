"""Validation patience is measured in epochs, including across resumes."""

import unittest

from configs import ConfigError, validate_config
from tests.test_training import _config
from training.early_stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def settings(self, **changes):
        return dict(enabled=True, monitor="edge_mAP", mode="max",
                    patience_epochs=50, min_delta=0.0, **changes)

    def test_fifty_epochs_not_fifty_checks(self):
        stop = EarlyStopping(self.settings())
        for epoch in range(5, 55, 5):
            self.assertFalse(stop.update(epoch, {"edge_mAP": 0.5}))
        self.assertTrue(stop.update(55, {"edge_mAP": 0.5}))
        self.assertEqual(stop.best_epoch, 5)

    def test_improvement_resets_patience(self):
        stop = EarlyStopping(self.settings())
        stop.update(5, {"edge_mAP": 0.5})
        self.assertFalse(stop.update(50, {"edge_mAP": 0.6}))
        self.assertFalse(stop.update(95, {"edge_mAP": 0.6}))
        self.assertTrue(stop.update(100, {"edge_mAP": 0.6}))

    def test_best_at_105_stops_at_155_without_minimum(self):
        settings = self.settings(min_epochs=0)
        stop = EarlyStopping(settings)
        stop.update(100, {"edge_mAP": 0.5})
        stop.update(105, {"edge_mAP": 0.6})
        resumed = EarlyStopping(settings, stop.state_dict())
        for epoch in range(110, 155, 5):
            self.assertFalse(resumed.update(epoch, {"edge_mAP": 0.59}))
        self.assertTrue(resumed.update(155, {"edge_mAP": 0.59}))
        self.assertEqual(resumed.best_epoch, 105)

    def test_minimum_200_epochs_gates_stopping_without_resetting_clock(self):
        settings = self.settings()
        settings["min_epochs"] = 200
        stop = EarlyStopping(settings)
        for epoch in range(5, 200, 5):
            self.assertFalse(stop.update(epoch, {"edge_mAP": 0.5}))
        self.assertTrue(stop.update(200, {"edge_mAP": 0.5}))

    def test_recent_improvement_extends_past_minimum_and_survives_resume(self):
        settings = self.settings()
        settings["min_epochs"] = 200
        stop = EarlyStopping(settings)
        stop.update(5, {"edge_mAP": 0.5})
        stop.update(180, {"edge_mAP": 0.6})
        resumed = EarlyStopping(settings, stop.state_dict())
        for epoch in range(185, 230, 5):
            self.assertFalse(resumed.update(epoch, {"edge_mAP": 0.6}))
        self.assertTrue(resumed.update(230, {"edge_mAP": 0.6}))

    def test_resume_keeps_elapsed_patience_and_terminal_state(self):
        settings = self.settings()
        stop = EarlyStopping(settings)
        stop.update(5, {"edge_mAP": 0.5})
        stop.update(45, {"edge_mAP": 0.4})
        resumed = EarlyStopping(settings, stop.state_dict())
        self.assertFalse(resumed.update(50, {"edge_mAP": 0.4}))
        self.assertTrue(resumed.update(55, {"edge_mAP": 0.4}))
        self.assertTrue(EarlyStopping(settings, resumed.state_dict()).stopped)

    def test_min_delta_and_min_mode(self):
        settings = self.settings()
        settings.update(monitor="validation_total", mode="min", min_delta=0.01)
        stop = EarlyStopping(settings)
        stop.update(5, {"validation_total": 1.0})
        stop.update(10, {"validation_total": 0.995})
        self.assertEqual(stop.best_epoch, 5)
        stop.update(15, {"validation_total": 0.98})
        self.assertEqual(stop.best_epoch, 15)

    def test_disabled_and_legacy_state(self):
        self.assertFalse(EarlyStopping().update(100, {}))
        stop = EarlyStopping(self.settings(), {})
        self.assertFalse(stop.update(105, {"edge_mAP": 0.5}))
        self.assertEqual(stop.best_epoch, 105)

    def test_invalid_observations_and_changed_resume_settings(self):
        stop = EarlyStopping(self.settings())
        with self.assertRaises(ValueError):
            stop.update(5, {})
        with self.assertRaises(FloatingPointError):
            stop.update(5, {"edge_mAP": float("nan")})
        stop.update(5, {"edge_mAP": 0.5})
        settings = self.settings()
        settings["monitor"] = "edge_f1"
        with self.assertRaises(ValueError):
            EarlyStopping(settings, stop.state_dict())

    def test_config_validation(self):
        config = _config()
        config["model"]["decoder"].update(hidden_dim=552, attention_heads=6)
        config["evaluation"]["training_metrics"]["enabled"] = True
        config["training"]["early_stopping"] = self.settings()
        validate_config(config)
        for key, value in (("patience_epochs", 0), ("min_delta", float("nan")),
                           ("mode", "min"), ("monitor", "typo"),
                           ("min_epochs", -1), ("min_epochs", 3)):
            with self.subTest(key=key):
                config["training"]["early_stopping"] = self.settings()
                config["training"]["early_stopping"][key] = value
                with self.assertRaises(ConfigError):
                    validate_config(config)
        config["training"]["early_stopping"] = self.settings()
        config["evaluation"]["training_metrics"]["enabled"] = False
        with self.assertRaises(ConfigError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
