from pathlib import Path

import requirements
import toml
from requirements.requirement import Requirement

# All req files to check
EXTRA_MAP = {
    "test": "tests/requirements.txt",
    "docs": "docs/requirements.txt",
}


def check_deps(deps):
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
    with open(folder.joinpath(requirement_file), "r") as f:
        _deps = [(req.name, req.specs) for req in requirements.parse(f)]

    req_deps = check_deps(_deps)

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
    toml_reqs = toml.load(folder.joinpath("pyproject.toml"))

    # install_requires
    _reqs = [Requirement.parse(_line) for _line in toml_reqs["project"]["dependencies"]]
    install_requires = check_deps([(req.name, req.specs) for req in _reqs])

    # extras
    extras_require = {
        k: [Requirement.parse(_line) for _line in lines]
        for k, lines in toml_reqs["project"]["optional-dependencies"].items()
    }
    extras_require = {k: check_deps([(req.name, req.specs) for req in _reqs]) for k, _reqs in extras_require.items()}

    # Resolve conflicts
    mismatches = {}
    mismatches["requirements.txt"] = get_conficts(install_requires, "requirements.txt")
    for extra_k, req_file in EXTRA_MAP.items():
        mismatches[req_file] = get_conficts(extras_require[extra_k], req_file)

    # Display the results
    if any(len(mismatch) > 0 for mismatch in mismatches.values()):
        mismatch_str = "version specifiers mismatches:\n"
        mismatch_str += "\n".join(
            f"- {lib}: {setup} (from setup.cfg) | {reqs} (from {req_file})"
            for req_file, issues in mismatches.items()
            for lib, setup, reqs in issues
        )
        raise AssertionError(mismatch_str)


if __name__ == "__main__":
    main()
