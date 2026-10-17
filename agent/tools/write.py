from langchain_core.tools import tool
import os
from pathlib import Path

# What the repo tree lists. Every node's prompt carries it, and on a large repo
# gitingest's full listing -- docs, fixtures, images, every test data file --
# was most of a 131k-token context on its own (django's ran a planning call to
# 127k before any research). The agent edits source, so the tree lists source.
_TREE_SOURCE_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd"}
_TREE_SKIP_DIRS = {"__pycache__", "node_modules", "build", "dist", "site-packages"}
# A backstop for a repo whose source alone is huge. ~3 chars per token for a
# box-drawn path listing, so the default is ~25k tokens.
_TREE_MAX_CHARS = int(os.getenv("SWE_TREE_MAX_CHARS", "80000"))


def _source_tree(root: Path) -> list[str]:
    """``root``'s directories and source files, as gitingest draws them.

    Sorted, so the listing is byte-identical on every run -- it sits near the
    head of every prompt, and a reordering would change every prefix. Files at
    the top level are kept whatever their type (``setup.py``, ``setup.cfg``,
    ``pyproject.toml`` say what the project is); below it, only source, and a
    directory with no source anywhere under it is left out.
    """

    def keep_dir(name: str) -> bool:
        return name not in _TREE_SKIP_DIRS and not name.endswith(".egg-info")

    def walk(directory: Path, top: bool) -> list[str]:
        """Lines for ``directory``'s contents, relative to it; empty if no source."""
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_dir(), p.name))
        except OSError:
            return []
        # (label, lines under it) -- None for a file.
        kept: list[tuple[str, list[str] | None]] = []
        for entry in entries:
            if entry.is_symlink() or entry.name.startswith("."):
                continue
            if entry.is_dir():
                if keep_dir(entry.name) and (below := walk(entry, False)):
                    kept.append((entry.name + "/", below))
            elif top or entry.suffix in _TREE_SOURCE_SUFFIXES:
                kept.append((entry.name, None))
        lines: list[str] = []
        for index, (label, below) in enumerate(kept):
            last = index == len(kept) - 1
            lines.append(("└── " if last else "├── ") + label)
            lines.extend(("    " if last else "│   ") + line for line in below or [])
        return lines

    return walk(root, True)


@tool(parse_docstring=True)
def create_file(path: str, content: str) -> str:
    """Create a new file at the specified path with the given content. If the file already exists,
    returns an error message instead of overwriting.

    Args:
        path: The path from the root of the folder to the file
        content: The text content to write to the file

    Returns:
        str: A success message with the file path, or an error message if creation failed
    """
    try:
        # Check if file already exists
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully created file at {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool(parse_docstring=True)
def write_to_file(path: str, content: str) -> str:
    """Override the content of an existing file at the specified path. If the file doesn't exist,
    returns an error message instead of creating it.

    Args:
        path: The path to the file to override
        content: The new text content that will completely replace the current content

    Returns:
        str: A success message with the file path, or an error message if writing failed
    """
    try:
        if not os.path.exists(path):
            return f"Error: File {path} does not exist"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully overridden file at {path}"
    except Exception as e:
        return f"Error overriding file: {str(e)}"


@tool(parse_docstring=True)
def get_files_structure(directory: str = "./workspace_repo") -> str:
    """Generate a JSON representation of the file and directory structure starting from the specified directory.
    Lists the source files, in the layout gitingest uses.

    Args:
        directory: The root directory to start scanning from (defaults to "./workspace_repo")

    Returns:
        str: A string representing the hierarchical directory structure and file listing
    """
    root = Path(directory)
    lines = _source_tree(root)
    tree = "Directory structure:\n" + f"└── {root.resolve().name}/\n" + "".join(f"    {line}\n" for line in lines)
    if len(tree) > _TREE_MAX_CHARS:
        # Cut at a line, and say so: a listing that silently stops reads as
        # the whole repo, and the agent would conclude the rest does not exist.
        cut = tree.rfind("\n", 0, _TREE_MAX_CHARS) + 1
        omitted = tree.count("\n", cut)
        tree = tree[:cut] + f"    ... ({omitted} more entries not shown; search the repo to find them)\n"
    return tree


# List of available tools
write_tools = [create_file, write_to_file, get_files_structure]
write_tools_map = {tool.name: tool for tool in write_tools}
