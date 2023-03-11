from dataclasses import dataclass
import os
from HistogramLib.store import AbstractHistogramId

@dataclass(unsafe_hash=True)
class HistogramId(AbstractHistogramId):
    clueParamName:str = ""
    datatype:str = ""
    histName:str = ""

    @property
    def path(self) -> str:
        """ get relative path from version folder (excluded) to pickle file name (included)"""
        return os.path.join(self.clueParamName, self.datatype, self.histName + ".pickle")