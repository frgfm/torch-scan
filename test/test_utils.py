import unittest
from torchscan import utils


class UtilsTester(unittest.TestCase):

    def test_format_name(self):
        name = 'mymodule'
        self.assertEqual(utils.format_name(name), name)
        self.assertEqual(utils.format_name(name, depth=1), f"├─{name}")
        self.assertEqual(utils.format_name(name, depth=3), f"|    |    └─{name}")

    def test_wrap_string(self):

        example = '.'.join(['a' for _ in range(10)])
        max_len = 10
        wrap = '[...]'

        self.assertEqual(utils.wrap_string(example, max_len, mode='end'),
                         example[:max_len - len(wrap)] + wrap)
        self.assertEqual(utils.wrap_string(example, max_len, mode='mid'),
                         f"{example[:max_len - 2 - len(wrap)]}{wrap}.a")
        self.assertEqual(utils.wrap_string(example, len(example), mode='end'), example)
        self.assertRaises(ValueError, utils.wrap_string, example, max_len, mode='test')

    def test_unit_scale(self):

        self.assertEqual(utils.unit_scale(3e14), (300, 'T'))
        self.assertEqual(utils.unit_scale(3e10), (30, 'G'))
        self.assertEqual(utils.unit_scale(3e7), (30, 'M'))
        self.assertEqual(utils.unit_scale(15e3), (15, 'k'))
        self.assertEqual(utils.unit_scale(500), (500, ''))


if __name__ == '__main__':
    unittest.main()
