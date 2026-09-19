"""CPU-only regression tests for the oracle and timeout evidence boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from survey import launch_process, selftest


class SurveyTests(unittest.TestCase):
    def test_oracle(self):
        selftest()

    def test_timeout_preserves_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); log = root / 'log.txt'; report_path = root / 'report.json'
            script = "import time; print('partial evidence', flush=True); time.sleep(30)"
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                launch_process([sys.executable, '-c', script], log, report_path,
                               {'pass': False}, timeout=1)
            report = json.loads(report_path.read_text())
            self.assertFalse(report['pass'])
            self.assertEqual(report['failure'], 'timeout')
            self.assertIsNone(report['exit_code'])
            self.assertIn('partial evidence', log.read_text())


if __name__ == '__main__':
    unittest.main()
