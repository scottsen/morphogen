"""
Operator decorator and metadata system.

This module provides the @operator decorator for marking domain operators
and attaching metadata (signatures, categories, documentation).
"""

from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass
from functools import wraps


class OpCategory(Enum):
    """Operator categories for semantic grouping."""
    CONSTRUCT = "construct"      # Create new data structures
    TRANSFORM = "transform"       # Transform existing data
    QUERY = "query"              # Query/analyze data
    INTEGRATE = "integrate"      # Time-stepping/integration
    COMPOSE = "compose"          # Combine multiple elements
    MUTATE = "mutate"            # In-place modification
    SAMPLE = "sample"            # Sample/resample data
    RENDER = "render"            # Render/visualize data


@dataclass
class OperatorMetadata:
    """Metadata attached to operator functions."""
    domain: str
    category: OpCategory
    signature: str
    deterministic: bool
    doc: str
    pure: bool = True              # No side effects
    stateful: bool = False         # Maintains internal state
    vectorized: bool = False       # Works on batches


def operator(
    domain: str,
    category: OpCategory,
    signature: str,
    deterministic: bool = True,
    doc: str = "",
    pure: bool = True,
    stateful: bool = False,
    vectorized: bool = False
) -> Callable:
    """
    Decorator to mark a function as a domain operator.

    Args:
        domain: Domain name (e.g., "graph", "audio", "field")
        category: Operator category (OpCategory enum)
        signature: Type signature (e.g., "(directed: bool) -> Graph")
        deterministic: Whether operator is deterministic
        doc: Documentation string
        pure: Whether operator has no side effects
        stateful: Whether operator maintains internal state
        vectorized: Whether operator works on batches

    Returns:
        Decorated function with _operator_metadata attribute

    Example:
        @operator(
            domain="graph",
            category=OpCategory.CONSTRUCT,
            signature="(directed: bool) -> Graph",
            deterministic=True,
            doc="Create an empty graph"
        )
        def create_empty(directed: bool = False) -> Graph:
            return Graph(directed=directed)
    """
    def decorator(func: Callable) -> Callable:
        # Attach metadata to function
        metadata = OperatorMetadata(
            domain=domain,
            category=category,
            signature=signature,
            deterministic=deterministic,
            doc=doc or func.__doc__ or "",
            pure=pure,
            stateful=stateful,
            vectorized=vectorized
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach metadata
        wrapper._operator_metadata = metadata

        return wrapper

    return decorator


def get_operator_metadata(func: Callable) -> Optional[OperatorMetadata]:
    """
    Get operator metadata from a function.

    Args:
        func: Function to inspect

    Returns:
        OperatorMetadata if function is an operator, None otherwise
    """
    return getattr(func, '_operator_metadata', None)


def is_operator(func: Callable) -> bool:
    """
    Check if a function is a domain operator.

    Args:
        func: Function to check

    Returns:
        True if function has operator metadata, False otherwise
    """
    return hasattr(func, '_operator_metadata')
