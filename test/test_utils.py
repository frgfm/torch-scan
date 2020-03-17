import unittest
from torchscan import utils


class UtilsTester(unittest.TestCase):

    def test_format_name(self):
        name = 'mymodule'
        self.assertEqual(utils.format_name(name), name)
        self.assertEqual(utils.format_name(name, depth=1), f"├─{name}")
        self.assertEqual(utils.format_name(name, depth=3), f"|    |    └─{name}")

    def test_wrap_string(self):

        string = '.'.join(['a' for _ in range(10)])
        max_len = 10
        wrap = '[...]'

        self.assertEqual(utils.wrap_string(string, max_len, mode='end'), string[:max_len - len(wrap)] + wrap)
        self.assertEqual(utils.wrap_string(string, max_len, mode='mid'), f"{string[:max_len - 2 - len(wrap)]}{wrap}.a")


if __name__ == '__main__':
    unittest.main()
