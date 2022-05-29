from pathlib import Path

import requirements
from requirements.requirement import Requirement

# All req files to check
EXTRA_MAP = {
    "test": "tests/requirements.txt",
    "docs": "docs/requirements.txt",
}
EXTRA_IGNORE = ["dev"]


def parse_deps(deps):
    reqs = {}
    for _dep in deps:
        lib, specs = _dep
        assert reqs.get(lib) is None, f"conflicting deps for {lib}"
        reqs[lib] = specs

    return reqs


def get_conficts(setup_reqs, requirement_file):

    # Parse the deps from the requirements.txt
    folder = Path(__file__).parent.parent.absolute()
    req_deps = {}
    with open(folder.joinpath(requirement_file), 'r') as f:
        _deps = [(req.name, req.specs) for req in requirements.parse(f)]

    req_deps = parse_deps(_deps)

    # Compare them
    assert len(req_deps) == len(setup_reqs)
    mismatches = []
    for k, v in setup_reqs.items():
        assert isinstance(req_deps.get(k), list)
        if req_deps[k] != v:
            mismatches.append((k, v, req_deps[k]))

    return mismatches


def main():

    # Collect the one from setup.py
    folder = Path(__file__).parent.parent.absolute()
    with open(folder.joinpath("setup.cfg"), 'r') as f:
        setup = f.readlines()

    # install_requires
    lines = setup[setup.index("install_requires =\n") + 1:]
    lines = [_dep.strip() for _dep in lines[:lines.index("\n")]]
    _reqs = [Requirement.parse(_line) for _line in lines]
    install_requires = parse_deps([(req.name, req.specs) for req in _reqs])

    # extras
    extras_require = {}
    lines = setup[setup.index("[options.extras_require]\n") + 1:]
    lines = lines[:lines.index("\n")]
    # Split each extra
    extra_lines = [_line for _line in lines if str.isalpha(_line[0])]
    extra_names = [_line.strip().replace("=", "").strip() for _line in extra_lines]
    for current_extra, start_line, end_line in zip(extra_names, extra_lines, extra_lines[1:] + [None]):
        if current_extra in EXTRA_IGNORE:
            continue
        _lines = [_dep for _dep in lines[lines.index(start_line) + 1:]]
        if isinstance(end_line, str):
            _lines = _lines[:_lines.index(end_line)]
        # Remove comments
        _lines = [_line.strip() for _line in _lines]
        _reqs = [Requirement.parse(_line.strip()) for _line in _lines if not _line.strip().startswith("#")]
        extras_require[current_extra] = parse_deps([(req.name, req.specs) for req in _reqs])

    # Resolve conflicts
    mismatches = {}
    mismatches["requirements.txt"] = get_conficts(install_requires, "requirements.txt")
    for extra_k, req_file in EXTRA_MAP.items():
        mismatches[req_file] = get_conficts(extras_require[extra_k], req_file)

    # Display the results
    if any(len(mismatch) > 0 for mismatch in mismatches.values()):
        mismatch_str = "version specifiers mismatches:\n"
        mismatch_str += '\n'.join(
            f"- {lib}: {setup} (from setup.cfg) | {reqs} (from {req_file})"
            for req_file, issues in mismatches.items() for lib, setup, reqs in issues
        )
        raise AssertionError(mismatch_str)

if __name__ == "__main__":
    main()
