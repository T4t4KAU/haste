import unittest

from haste.utils.profiling import build_profile_report, summarize_numeric_series


class ProfilingUtilsTest(unittest.TestCase):
    def test_summarize_numeric_series_reports_percentiles(self):
        summary = summarize_numeric_series([1, 2, 3, 4])

        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 2.5)
        self.assertAlmostEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)

    def test_build_profile_report_computes_core_throughput_and_acceptance(self):
        metrics = {
            "prefill_total_tokens": 100,
            "decode_total_tokens": 50,
            "prefill_total_time": 2.0,
            "decode_total_time": 5.0,
            "target_step_times": [1.0, 2.0],
            "scheduler_times": [0.1, 0.2],
            "prefill_step_times": [2.1],
            "decode_step_times": [1.5, 3.5],
            "prefill_batch_sizes": [4],
            "decode_batch_sizes": [4, 2],
            "prefill_step_tokens": [100],
            "decode_step_tokens": [20, 30],
            "prefill_speculator_times": [0.3],
            "prefill_verifier_times": [1.7],
            "speculate_times": [0.4, 0.5],
            "verify_times": [0.8, 0.9],
            "rollback_times": [0.05, 0.04],
            "postprocess_times": [0.03, 0.02],
            "target_verify_times": [0.7, 0.8],
            "cache_hits": [0.25, 0.75],
            "accepted_suffix_lens_with_recovery": [3, 4],
            "accepted_suffix_lens_on_hit": [4],
            "accepted_suffix_lens_on_miss": [3],
            "num_requests": 2,
            "completed_requests": 2,
            "num_engine_steps": 2,
            "runner_profiles": {"target": {"device": "cpu"}},
        }

        report = build_profile_report(
            metrics,
            wall_time_sec=10.0,
            generated_new_tokens=20,
            requested_new_tokens=24,
            speculate_k=7,
        )

        self.assertAlmostEqual(report["throughput"]["prefill_tok_per_s"], 50.0)
        self.assertAlmostEqual(report["throughput"]["decode_tok_per_s"], 10.0)
        self.assertAlmostEqual(report["throughput"]["overall_tok_per_s"], 15.0)
        self.assertAlmostEqual(report["cache"]["avg_hit_rate"], 0.5)
        self.assertAlmostEqual(report["acceptance"]["avg_tokens_per_step_with_recovery"], 3.5)
        self.assertAlmostEqual(report["acceptance"]["avg_accepted_spec_tokens"], 2.5)
        self.assertEqual(report["runners"]["target"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
