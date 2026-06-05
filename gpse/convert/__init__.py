"""
GPSE convert package — data conversion and genotype QC utilities.

This package provides tools for converting between common genomic data
formats, running quality-control filters, and preparing genotype matrices
for downstream GPSE analyses.
"""

from .processor import GenomicDataProcessor

__all__ = ["GenomicDataProcessor"]
