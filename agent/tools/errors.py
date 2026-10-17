def tool_error_message(error: Exception) -> str:
    """What the model sees when a research tool raises.

    Handed to ``ToolNode(handle_tool_errors=...)`` so a failed call becomes a
    message the model can act on instead of an exception that ends the
    instance. ``str(error)`` rather than ToolNode's default ``repr``: for the
    common case, a guessed path, only the former names the file.
    """
    return (
        f"Error: {type(error).__name__}: {error}\n"
        "Check the path against the repository structure, or search for the file, and try again."
    )
