import pkg_resources
from .h5sparse import Group, File, Dataset, get_format_str  # noqa: F401

__version__ = pkg_resources.get_distribution("h5sparse-tensor").version
