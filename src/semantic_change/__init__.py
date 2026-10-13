"""Semantic Change Detection Engine.

Compares two versions of a text and reports the changes in meaning. The pipeline has
five stages: clause segmentation (spaCy), sentence embeddings (sentence-transformers),
bipartite alignment (SciPy), an evaluation agent (Claude) and a JSON and Markdown
output. ``semantic_change.api`` serves the pipeline as a REST API.
"""

__version__ = "2.0.0"
