"""PRISM Pipelines — Dataset Ingestion Pipeline"""

from pipelines.ingestion.base_adapter import BaseAdapter
from pipelines.ingestion.coswara_adapter import CoswaraAdapter
from pipelines.ingestion.coughvid_adapter import CoughvidAdapter
from pipelines.ingestion.icbhi_adapter import IcbhiAdapter

__all__ = ["BaseAdapter", "CoughvidAdapter", "CoswaraAdapter", "IcbhiAdapter"]
