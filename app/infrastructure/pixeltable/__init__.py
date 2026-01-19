"""Pixeltable infrastructure package initialization."""

from app.infrastructure.pixeltable.client import (
    PixeltableClient,
    get_pixeltable_client,
    init_pixeltable,
)
from app.infrastructure.pixeltable.tables import (
    PixeltableSchemas,
    get_schemas,
    init_tables,
)
from app.infrastructure.pixeltable.udfs import (
    exact_match,
    substring_match,
    string_similarity,
    correctness_metric,
    quality_metric,
    register_all_udfs,
    get_metric_function,
    METRIC_INFO,
)

__all__ = [
    "PixeltableClient",
    "get_pixeltable_client",
    "init_pixeltable",
    "PixeltableSchemas",
    "get_schemas",
    "init_tables",
    # UDFs
    "exact_match",
    "substring_match",
    "string_similarity",
    "correctness_metric",
    "quality_metric",
    "register_all_udfs",
    "get_metric_function",
    "METRIC_INFO",
]
