"""CPU checks for frozen parent generation, unique pair sampling and gate math."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('workloads', Path(__file__).parent/'src/workloads.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


class WorkloadTests(unittest.TestCase):
    def test_corrected_reference_keeps_counterexample(self):
        from src.cpu_reference import terminal_order_distances
        x = np.load(Path(__file__).parent.parent/'lite_selector_20260920/results/g0_failure_subset_v1.npy', allow_pickle=False)
        distance = terminal_order_distances(x)
        self.assertEqual(float(distance[25,28]).hex(),'0x1.74120418f0f3fp-4')
        threshold = float.fromhex('0x1.74120418f0f3ep-4')
        self.assertEqual(int(np.count_nonzero(distance <= threshold)),34)
        self.assertEqual(int(np.count_nonzero(distance <= np.nextafter(threshold,np.inf))),36)

    def test_unique_unordered_pairs(self):
        i,j = w.sample_pairs(1024)
        self.assertEqual(len(i),65536)
        self.assertTrue(np.all(i<j)); self.assertTrue(np.all(j<1024))
        self.assertEqual(len(np.unique(i*1024+j)),65536)
        ii,jj = w.sample_pairs(1024)
        np.testing.assert_array_equal(i,ii); np.testing.assert_array_equal(j,jj)

    def test_parents(self):
        for family in w.FAMILIES:
            x=w.generate(family,101); y=w.generate(family,101)
            self.assertEqual(x.shape,(4096,512)); self.assertEqual(x.dtype,np.float32)
            self.assertTrue(x.flags.c_contiguous); self.assertLessEqual(np.abs(x).max(),1)
            np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(x[:1024],y[:1024])
        all_seeds=sum(w.SPLITS.values(),[])
        self.assertEqual(len(set(all_seeds)),len(all_seeds))



class EstimatorTests(unittest.TestCase):
    def test_ratio_direction_and_interval(self):
        spec=importlib.util.spec_from_file_location('gate_summary',Path(__file__).parent/'summarize.py')
        s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
        self.assertAlmostEqual(s.gm([1,4]),2)
        out=s.interval([np.log(2.)]*4)
        self.assertAlmostEqual(out['geomean'],2)
        self.assertEqual(out['log_t95_interval'],[2.,2.])
        self.assertAlmostEqual(s.quantile([1,2,3,4],.95),3.85)

    def test_equal_parent_weight(self):
        spec=importlib.util.spec_from_file_location('gate_summary',Path(__file__).parent/'summarize.py')
        s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
        rows=[{'parent_id':parent,'process_medians_ms':{'a':[ratio]*4,'b':[1]*4}}
              for parent,ratio in [('x',2),('y',.5)] for _ in range(4)]
        self.assertAlmostEqual(s.grouped_ratio(rows,'a','b')['geomean'],1)

if __name__ == '__main__': unittest.main()
