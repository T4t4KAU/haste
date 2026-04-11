import threading
import unittest

import torch

from haste.utils.context import get_context, reset_context, set_context


class ContextThreadLocalTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_context_is_thread_local(self):
        results = {}

        set_context(is_prefill=False, context_lens=torch.tensor([11], dtype=torch.int32))

        def worker():
            set_context(is_prefill=True, context_lens=torch.tensor([7], dtype=torch.int32))
            results["worker"] = int(get_context().context_lens[0].item())
            reset_context()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertEqual(results["worker"], 7)
        self.assertEqual(int(get_context().context_lens[0].item()), 11)


if __name__ == "__main__":
    unittest.main()
