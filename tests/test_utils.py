import pytest

from torchscan import utils


def test_format_name():
    name = "mymodule"
    assert utils.format_name(name) == name
    assert utils.format_name(name, depth=1) == f"├─{name}"
    assert utils.format_name(name, depth=3) == f"|    |    └─{name}"


def test_wrap_string():
    example = ".".join(["a" for _ in range(10)])
    max_len = 10
    wrap = "[...]"

    assert utils.wrap_string(example, max_len, mode="end") == example[: max_len - len(wrap)] + wrap
    assert utils.wrap_string(example, max_len, mode="mid") == f"{example[:max_len - 2 - len(wrap)]}{wrap}.a"
    assert utils.wrap_string(example, len(example), mode="end") == example
    with pytest.raises(ValueError):
        _ = utils.wrap_string(example, max_len, mode="test")


@pytest.mark.parametrize(
    ("input_val", "num_val", "unit"),
    [
        (3e14, 300, "T"),
        (3e10, 30, "G"),
        (3e7, 30, "M"),
        (15e3, 15, "k"),
        (500, 500, ""),
    ],
)
def test_unit_scale(input_val, num_val, unit):
    assert utils.unit_scale(input_val) == (num_val, unit)
