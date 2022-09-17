"""Package-wide helping functions for setup."""
import importlib
import sys


def str2bool(value: str) -> bool:
    """Convert a string to its boolean representation.

    False valid values: 'false', 'f', '0', 'no', and 'n'.
    True valid values: 'true', 't', '1', 'yes', and 'y'.

    Args:
        value (str): String representing a boolean value.

    Raises:
        ValueError: Failed to parse value.

    Returns:
        bool: Boolean representation of the value.
    """
    value = value.lower()

    if value not in {"false", "f", "0", "no", "n", "true", "t", "1", "yes", "y"}:
        raise ValueError(f"{value} is not a valid boolean value")

    if isinstance(value, bool):
        out = value
    if value in {"false", "f", "0", "no", "n"}:
        out = False
    elif value in {"true", "t", "1", "yes", "y"}:
        out = True

    return out


def import_module(module_path):
    """(Re)Import a Python module, as for a config file."""
    try:
        return importlib.reload(sys.modules[module_path.stem])
    except KeyError:
        pass

    assert module_path.exists(), f"File {module_path} does not exist."
    if module_path.parent not in sys.path:
        sys.path.append(str(module_path.parent.resolve()))
    return importlib.import_module(module_path.stem)


def requirements(infile):
    """Parse pip-formatted requirements file."""
    with open(infile) as f:
        packages = f.read().splitlines()
    result = []
    for pkg in packages:
        if pkg.strip().startswith("#") or not pkg.strip():
            continue
        result.append(pkg)
    return result
