import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server as haste_server


class _TokenizerWithTemplate:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return " | ".join(f"{message['role']}:{message['content']}" for message in messages) + " | assistant:"


class _TokenizerWithoutTemplate:
    pass


class _TokenizerMissingTemplateConfig:
    def apply_chat_template(self, messages, **kwargs):
        raise ValueError(
            "Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!"
        )


class ServerHelpersTest(unittest.TestCase):
    def test_parse_prompt_inputs_supports_single_prompt(self):
        prompts, is_single = haste_server.parse_prompt_inputs({"prompt": "hello"})

        self.assertEqual(prompts, ["hello"])
        self.assertTrue(is_single)

    def test_parse_prompt_inputs_supports_token_batch(self):
        prompts, is_single = haste_server.parse_prompt_inputs({"prompt_token_ids_batch": [[1, 2], [3, 4]]})

        self.assertEqual(prompts, [[1, 2], [3, 4]])
        self.assertFalse(is_single)

    def test_build_sampling_params_list_repeats_single_payload(self):
        params = haste_server.build_sampling_params_list(
            {
                "temperature": 0.2,
                "draft_temperature": 0.4,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
            count=2,
            default_max_new_tokens=8,
        )

        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].temperature, 0.2)
        self.assertEqual(params[0].draft_temperature, 0.4)
        self.assertEqual(params[0].max_new_tokens, 16)
        self.assertTrue(params[0].ignore_eos)

    def test_build_sampling_params_list_supports_per_prompt_list(self):
        params = haste_server.build_sampling_params_list(
            {
                "sampling_params": [
                    {"temperature": 0.0, "max_new_tokens": 4},
                    {"temperature": 0.7, "max_tokens": 6},
                ]
            },
            count=2,
            default_max_new_tokens=8,
        )

        self.assertEqual([item.temperature for item in params], [0.0, 0.7])
        self.assertEqual([item.max_new_tokens for item in params], [4, 6])

    def test_render_chat_prompt_prefers_chat_template(self):
        prompt = haste_server.render_chat_prompt(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            _TokenizerWithTemplate(),
        )

        self.assertIn("system:You are helpful.", prompt)
        self.assertTrue(prompt.endswith("assistant:"))

    def test_render_chat_prompt_falls_back_without_chat_template(self):
        prompt = haste_server.render_chat_prompt(
            [
                {"role": "user", "content": "Hello"},
            ],
            _TokenizerWithoutTemplate(),
        )

        self.assertEqual(prompt, "USER: Hello\nASSISTANT:")

    def test_render_chat_prompt_falls_back_when_chat_template_missing(self):
        prompt = haste_server.render_chat_prompt(
            [
                {"role": "user", "content": "Hello"},
            ],
            _TokenizerMissingTemplateConfig(),
        )

        self.assertEqual(prompt, "USER: Hello\nASSISTANT:")

    def test_strip_thinking_output_removes_think_block(self):
        text = "<think>\ninternal reasoning\n</think>\n\nFinal answer."

        cleaned = haste_server.strip_thinking_output(text)

        self.assertEqual(cleaned, "Final answer.")

    def test_strip_thinking_output_drops_dangling_think_only_output(self):
        text = "<think>\ninternal reasoning without final answer"

        cleaned = haste_server.strip_thinking_output(text)

        self.assertEqual(cleaned, "")

    def test_strip_thinking_output_removes_special_chat_tokens(self):
        text = "2 + 2 = 4.<|im_end|>"

        cleaned = haste_server.strip_thinking_output(text)

        self.assertEqual(cleaned, "2 + 2 = 4.")


if __name__ == "__main__":
    unittest.main()
