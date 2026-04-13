from threading import Lock
from types import SimpleNamespace
import unittest

from haste.engine.draft_runner import AutoTuneState, DraftRunner


class AutoTunePolicyTest(unittest.TestCase):
    def _make_runner(self, *, auto_tune: bool) -> DraftRunner:
        runner = DraftRunner.__new__(DraftRunner)
        runner.config = SimpleNamespace(
            async_auto_tune=auto_tune,
            speculate_k=7,
            async_fan_out=3,
            async_auto_tune_min_k=1,
            async_auto_tune_max_k=7,
            async_auto_tune_min_f=1,
            async_auto_tune_max_f=3,
            async_auto_tune_probe_steps=2,
            async_auto_tune_reprobe_interval=8,
            async_auto_tune_margin=0.95,
            async_auto_tune_wait_ratio=0.08,
            async_auto_tune_underfill_ratio=0.55,
            async_auto_tune_accept_floor=0.35,
            async_auto_tune_cache_hit_target=0.7,
            async_auto_tune_ema_alpha=0.35,
            async_auto_tune_score_tolerance=0.05,
            verbose=False,
        )
        runner._controller_lock = Lock()
        runner._worker_profile = {
            "cache_populate_times": [0.02],
            "request_wait_times": [],
            "exposed_wait_times": [],
        }
        runner._last_request_wait_ms = 0.0
        runner._last_request_serve_ms = 10.0
        runner._last_exposed_wait_ms = 0.0
        runner._last_cache_hit_rate = 0.2
        runner._last_accept_fraction = 0.0
        runner._fan_out_batch_hint = 0
        runner._runtime_lookahead_cap = 1
        runner._runtime_fan_out_cap = 1
        runner._auto_tune_state = None
        runner._last_logged_policy = None
        return runner

    def test_auto_tune_search_starts_from_small_k_and_grows_when_hidden(self):
        runner = self._make_runner(auto_tune=True)
        runner._auto_tune_state = AutoTuneState(
            stage="search_k",
            trial_k=1,
            trial_f=1,
            settled_k=1,
            settled_f=1,
            best_hidden_k=1,
            best_hidden_f=1,
        )

        runner.report_verify_feedback(verify_elapsed_s=0.10, batch_size=8, accepted_fraction=0.6)
        self.assertEqual(runner._runtime_lookahead_cap, 1)
        self.assertEqual(runner._auto_tune_state.trial_observations, 1)

        runner.report_verify_feedback(verify_elapsed_s=0.10, batch_size=8, accepted_fraction=0.6)
        self.assertEqual(runner._runtime_lookahead_cap, 2)
        self.assertEqual(runner._runtime_fan_out_cap, 1)
        self.assertEqual(runner._auto_tune_state.stage, "search_k")
        self.assertEqual(runner._auto_tune_state.trial_k, 2)

    def test_static_mode_keeps_user_caps(self):
        runner = self._make_runner(auto_tune=False)
        runner._runtime_lookahead_cap = 7
        runner._runtime_fan_out_cap = 3

        runner.report_verify_feedback(verify_elapsed_s=0.10, batch_size=8, accepted_fraction=0.6)

        self.assertEqual(runner._runtime_lookahead_cap, 7)
        self.assertEqual(runner._runtime_fan_out_cap, 3)

    def test_steady_state_grows_k_when_underfilled(self):
        runner = self._make_runner(auto_tune=True)
        runner._runtime_lookahead_cap = 2
        runner._runtime_fan_out_cap = 1
        runner._auto_tune_state = AutoTuneState(
            stage="steady",
            trial_k=2,
            trial_f=1,
            settled_k=2,
            settled_f=1,
            best_hidden_k=2,
            best_hidden_f=1,
            stable_steps=runner.config.async_auto_tune_reprobe_interval - 1,
            ema_verify_ms=100.0,
            ema_populate_ms=20.0,
            ema_wait_ms=15.0,
            ema_serve_ms=15.0,
            ema_exposed_ms=0.0,
            ema_cache_hit_rate=0.3,
            ema_accept_fraction=0.6,
            trial_observations=runner.config.async_auto_tune_probe_steps,
        )
        runner._worker_profile["cache_populate_times"].append(0.02)
        runner._last_request_wait_ms = 15.0
        runner._last_request_serve_ms = 15.0
        runner._last_exposed_wait_ms = 0.0

        runner.report_verify_feedback(verify_elapsed_s=0.10, batch_size=8, accepted_fraction=0.6)

        self.assertEqual(runner._runtime_lookahead_cap, 3)
        self.assertEqual(runner._runtime_fan_out_cap, 1)


if __name__ == "__main__":
    unittest.main()
