"""Small CPU checks for the direction and pairing of the timing estimator."""
import math
import unittest

from summarize import ratio_stats


class SummaryTests(unittest.TestCase):
    def test_ratio_direction(self):
        row = ratio_stats([2, 4, 6, 8], [1, 2, 3, 4])
        self.assertAlmostEqual(row['geomean'], 2)
        self.assertEqual(row['process_wins'], 4)
        self.assertTrue(row['all_processes_gt_1_10'])
        self.assertEqual(row['log_t95_interval'], [2, 2])

    def test_not_a_ratio_of_marginals(self):
        row = ratio_stats([2, 100, 2, 100], [1, 200, 1, 200])
        self.assertAlmostEqual(row['geomean'], 1)
        self.assertAlmostEqual(row['forward_geomean'], 2)
        self.assertAlmostEqual(row['reverse_geomean'], .5)
        self.assertEqual(row['process_wins'], 2)
        self.assertFalse(row['all_processes_gt_1_10'])
        lo, hi = row['log_t95_interval']
        self.assertLess(lo, 1); self.assertGreater(hi, 1)
        self.assertTrue(math.isclose(lo * hi, 1))

    def test_strict_gate(self):
        self.assertFalse(ratio_stats([1.1] * 4, [1] * 4)['all_processes_gt_1_10'])


if __name__ == '__main__':
    unittest.main()
