"""Host-side regression tests for optional backend dispatch and state isolation."""
import unittest
from concurrent.futures import ThreadPoolExecutor
import time

import torch
import submission as mod


class Backend:
    def __init__(self):
        self.sms = 83
        self.deterministic = False
        self.fail = False
        self.calls = 0

    def get_num_sms(self):
        return self.sms

    def set_num_sms(self, value):
        self.sms = value

    def get_deterministic_algorithms(self):
        return self.deterministic

    def use_deterministic_algorithms(self, value):
        self.deterministic = value

    def bf16_gemm_nt(self, x, w, out):
        expected = self.sms
        self.calls += 1
        time.sleep(0.005)  # Release the GIL, like a native FFI call can.
        assert self.sms == expected and self.deterministic
        if self.fail:
            raise RuntimeError("injected backend failure")
        out.copy_(torch.mm(x, w.T))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.state = {"qkv_gate": torch.randn(16, 16).bfloat16(),
                      "transposed": {}, "sms": 132,
                      "deep": self.backend}
        self.x = torch.randn(2, 16).bfloat16()

    def call(self, sms=128, x=None):
        if x is not None:
            from unittest.mock import patch
            with patch.dict(mod._TACTICS, {("qkv_gate", 2, 16, 16): ("deep", sms)}):
                return mod.kernel_fn({"x": x, "config": {"layer": "qkv_gate", "n": 16}}, self.state)
        return mod._run(self.x, "qkv_gate", 16, self.state, ("deep", sms))

    def assert_restored(self):
        self.assertEqual(self.backend.sms, 83)
        self.assertFalse(self.backend.deterministic)

    def test_success_restores_callers_settings(self):
        torch.testing.assert_close(self.call(), self.x @ self.state["qkv_gate"].T)
        self.assert_restored()

    def test_exception_restores_callers_settings(self):
        self.backend.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.call()
        self.assert_restored()

    def test_concurrent_calls_serialize_configuration_and_launch(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.call, [128, 120, 132, 100] * 3))
        self.assertEqual(self.backend.calls, 12)
        self.assert_restored()

    def test_unsupported_layouts_bypass_native_backend(self):
        padded = torch.empty(2, 17, dtype=torch.bfloat16)[:, :16]
        offset = torch.empty(33, dtype=torch.bfloat16)[1:].view(2, 16)
        for x in (padded, offset):
            x.copy_(self.x)
            torch.testing.assert_close(self.call(x=x), self.x @ self.state["qkv_gate"].T)
        self.assertEqual(self.backend.calls, 0)
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
