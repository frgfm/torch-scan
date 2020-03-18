import unittest
import io
import sys
import torch.nn as nn
from torchscan import crawler


class UtilsTester(unittest.TestCase):

    def test_apply(self):
        multi_convs = nn.Sequential(nn.Conv2d(16, 32, 3), nn.Conv2d(32, 64, 3))
        mod = nn.Sequential(nn.Conv2d(3, 16, 3), multi_convs)

        # Tag module attributes
        def tag_name(mod, depth, name):
            mod.__depth__ = depth
            mod.__name__ = name

        crawler.apply(mod, tag_name)

        self.assertEqual(mod[1][1].__depth__, 2)
        self.assertEqual(mod[1][1].__name__, '1')

    def test_crawl_module(self):

        mod = nn.Conv2d(3, 8, 3)

        res = crawler.crawl_module(mod, (3, 32, 32))
        self.assertIsInstance(res, dict)
        self.assertEqual(res['overall']['grad_params'], 224)
        self.assertEqual(res['layers'][0]['output_shape'], (-1, 8, 30, 30))

    def test_summary(self):

        mod = nn.Conv2d(3, 8, 3)

        # Redirect stdout with StringIO object
        captured_output = io.StringIO()
        sys.stdout = captured_output
        crawler.summary(mod, (3, 32, 32))
        # Reset redirect.
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().split('\n')[7], 'Total params: 224')


if __name__ == '__main__':
    unittest.main()
