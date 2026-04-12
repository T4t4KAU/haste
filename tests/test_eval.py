import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval as eval_module


class EvalHelpersTest(unittest.TestCase):
    def test_select_judge_prompt_name_prefers_math_template_with_reference(self):
        sample = {
            "turns": ["What is 1+1?"],
            "reference": ["2"],
            "category": "math",
        }

        prompt_name = eval_module.select_judge_prompt_name(sample, dataset_name="gsm8k")

        self.assertEqual(prompt_name, "single-math-v1")

    def test_select_judge_prompt_name_uses_multi_turn_general_template(self):
        sample = {
            "turns": ["Q1", "Q2"],
            "category": "writing",
        }

        prompt_name = eval_module.select_judge_prompt_name(sample, dataset_name="mt_bench")

        self.assertEqual(prompt_name, "single-v1-multi-turn")

    def test_build_judge_prompt_single_math(self):
        sample = {
            "turns": ["Solve 2+2"],
            "reference": ["4"],
        }
        prompt_record = {
            "name": "single-math-v1",
            "system_prompt": "judge",
            "prompt_template": "Q={question} REF={ref_answer_1} A={answer}",
        }

        system_prompt, prompt = eval_module.build_judge_prompt(
            sample=sample,
            answers=["It is 4."],
            prompt_record=prompt_record,
        )

        self.assertEqual(system_prompt, "judge")
        self.assertEqual(prompt, "Q=Solve 2+2 REF=4 A=It is 4.")

    def test_build_judge_prompt_multi_turn_general(self):
        sample = {
            "turns": ["Q1", "Q2"],
        }
        prompt_record = {
            "name": "single-v1-multi-turn",
            "system_prompt": "judge",
            "prompt_template": "{question_1}|{answer_1}|{question_2}|{answer_2}",
        }

        _, prompt = eval_module.build_judge_prompt(
            sample=sample,
            answers=["A1", "A2"],
            prompt_record=prompt_record,
        )

        self.assertEqual(prompt, "Q1|A1|Q2|A2")

    def test_parse_judge_score_extracts_marker(self):
        self.assertEqual(eval_module.parse_judge_score("Good answer. Rating: [[8]]"), 8)

    def test_summarize_results_builds_accuracy(self):
        summary = eval_module.summarize_results(
            [
                {"dataset": "alpaca", "judge_score": 8},
                {"dataset": "alpaca", "judge_score": 6},
                {"dataset": "gsm8k", "judge_score": 9},
            ],
            pass_score=7,
        )

        self.assertEqual(summary["num_samples"], 3)
        self.assertAlmostEqual(summary["average_score"], (8 + 6 + 9) / 3)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertIn("alpaca", summary["per_dataset"])
        self.assertEqual(summary["score_distribution"]["8"], 1)


if __name__ == "__main__":
    unittest.main()
