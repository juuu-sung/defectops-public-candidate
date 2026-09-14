import unittest

from defectops.padim_benchmark import latency_summary


class PadimBenchmarkTests(unittest.TestCase):
    def test_latency_summary_uses_higher_p95(self):
        result = latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["p95_higher"], 4.0)

    def test_latency_summary_rejects_empty_or_negative_values(self):
        with self.assertRaises(ValueError):
            latency_summary([])
        with self.assertRaises(ValueError):
            latency_summary([1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
