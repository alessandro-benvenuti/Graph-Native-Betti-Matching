"""Epoch-based validation patience, with explicitly resumable state."""

import math


class EarlyStopping:
    def __init__(self, config=None, state=None):
        config = config or {}
        self.config = {
            "enabled": bool(config.get("enabled", False)),
            "monitor": config.get("monitor", "edge_mAP"),
            "mode": config.get("mode", "max"),
            "patience_epochs": int(config.get("patience_epochs", 50)),
            "min_epochs": int(config.get("min_epochs", 0)),
            "min_delta": float(config.get("min_delta", 0.0)),
        }
        self.best = None
        self.best_epoch = None
        self.last_epoch = None
        self.stopped = False
        if self.config["enabled"] and state and state["config"]["enabled"]:
            if {"min_epochs": 0, **state["config"]} != self.config:
                raise ValueError("Early-stopping configuration changed during resume")
            self.best = state["best"]
            self.best_epoch = state["best_epoch"]
            self.last_epoch = state["last_epoch"]
            self.stopped = bool(state["stopped"])

    def update(self, epoch, metrics):
        if not self.config["enabled"]:
            return False
        if self.stopped:
            return True
        if self.last_epoch is not None and epoch <= self.last_epoch:
            raise ValueError("Early stopping requires increasing validation epochs")
        name = self.config["monitor"]
        if name not in metrics or metrics[name] is None:
            raise ValueError("Early-stopping metric missing: " + name)
        value = float(metrics[name])
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite early-stopping metric: " + name)
        delta = self.config["min_delta"]
        improved = self.best is None or (
            value > self.best + delta if self.config["mode"] == "max"
            else value < self.best - delta
        )
        if improved:
            self.best, self.best_epoch = value, int(epoch)
        self.last_epoch = int(epoch)
        self.stopped = (
            epoch >= self.config["min_epochs"]
            and epoch - self.best_epoch >= self.config["patience_epochs"]
        )
        return self.stopped

    def state_dict(self):
        return {
            "config": dict(self.config), "best": self.best,
            "best_epoch": self.best_epoch, "last_epoch": self.last_epoch,
            "stopped": self.stopped,
        }
