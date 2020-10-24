# -*- coding: utf-8 -*-

import io
import sys
import unittest

import torch.nn as nn

from torchscan import crawler


class UtilsTester(unittest.TestCase):
    def test_apply(self):
        multi_convs = nn.Sequential(nn.Conv2d(16, 32, 3), nn.Conv2d(32, 64, 3))
        mod = nn.Sequential(nn.Conv2d(3, 16, 3), multi_convs)

        # Tag module attributes
        def tag_name(mod, name):
            mod.__depth__ = len(name.split('.')) - 1
            mod.__name__ = name.rpartition('.')[-1]

        crawler.apply(mod, tag_name)

        self.assertEqual(mod[1][1].__depth__, 2)
        self.assertEqual(mod[1][1].__name__, '1')

    def test_crawl_module(self):

        mod = nn.Conv2d(3, 8, 3)

        res = crawler.crawl_module(mod, (3, 32, 32))
        self.assertIsInstance(res, dict)
        self.assertEqual(res['overall']['grad_params'], 224)
        self.assertEqual(res['layers'][0]['output_shape'], (-1, 8, 30, 30))
        # Check receptive field order
        mod = nn.Sequential(nn.Conv2d(3, 8, 5), nn.ReLU(), nn.Conv2d(8, 16, 3))
        inv_res = crawler.crawl_module(mod, (3, 32, 32), relative_to_input=False)
        inv_order = [(_layer['rf'], _layer['s'], _layer['p']) for _layer in res['layers']]
        res = crawler.crawl_module(mod, (3, 32, 32), relative_to_input=True)
        order = [(_layer['rf'], _layer['s'], _layer['p']) for _layer in inv_res['layers']]
        self.assertTrue(all(x == y for x, y in zip(order[-1], inv_order[0])))

    def test_summary(self):

        mod = nn.Conv2d(3, 8, 3)

        # Redirect stdout with StringIO object
        captured_output = io.StringIO()
        sys.stdout = captured_output
        crawler.summary(mod, (3, 32, 32))
        # Reset redirect.
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().split('\n')[7], 'Total params: 224')

        # Check receptive field
        captured_output = io.StringIO()
        sys.stdout = captured_output
        crawler.summary(mod, (3, 32, 32), receptive_field=True)
        # Reset redirect.
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().split('\n')[1].rpartition('  ')[-1], 'Receptive field')
        self.assertEqual(captured_output.getvalue().split('\n')[3].split()[-1], '3')


if __name__ == '__main__':
    unittest.main()
