import glob
from os.path import expanduser, expandvars
from typing import Union

from omegaconf.listconfig import ListConfig


def glob_all(globs: Union[str, list[str], ListConfig]) -> list[str]:
    """
    Expand a list of glob strings into a list of file paths.

    Args:
        globs: A glob string or a list of glob strings.

    Returns:
        A list of file paths matching the glob strings.
    """

    if isinstance(globs, str):
        globs = [globs]
    elif isinstance(globs, ListConfig):
        globs = list(globs)
    assert isinstance(globs, list)

    files = []
    for glob_str in globs:
        glob_str = expandvars(expanduser(glob_str))
        files.extend(glob.glob(glob_str))

    return sorted(files)
