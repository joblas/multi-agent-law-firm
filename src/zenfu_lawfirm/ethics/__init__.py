"""
Ethics module for the ZenFu Law Firm AI system.
This module handles ethical considerations including bias detection, transparency, and accountability.
"""

from .bias_detector import BiasDetector
from .transparency import TransparencyManager
from .accountability import AccountabilityTracker

__all__ = ['BiasDetector', 'TransparencyManager', 'AccountabilityTracker']
