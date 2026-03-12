from dataclasses import dataclass
from typing import Dict


@dataclass
class Cell:
    cid: int
    ctype: int
    genotype: int = 0
    r: float = 0.0
    p: float = 0.0
    q: float = 0.0
    p_hold: int = 0


def count_cells_by_type(cells: Dict[int, Cell], OPC: int, OLIGO: int, MICRO: int):
    n_opc = sum(1 for c in cells.values() if c.ctype == OPC)
    n_ol = sum(1 for c in cells.values() if c.ctype == OLIGO)
    n_mg = sum(1 for c in cells.values() if c.ctype == MICRO)
    return n_opc, n_ol, n_mg