"""Layer 2 — data standard & capture (doc §5).

The open standard for a captured intervention: a time-synchronized log of device
kinematics + the paired Layer-1 observation + outcome (`schema.Episode`), plus the
machinery to produce it from the simulator (`capture`), iterate a corpus
(`replay`), and close the §3.6 calibration loop on it (`calibrate`). The real
patient corpus is the proprietary moat (§327) and stays private behind the same
`Episode` seam — never in this repo (firewall: provenance == "procedural").
"""

from lumen.data.schema import (SCHEMA_VERSION, Episode, EpisodeMeta, Outcome, Step,
                               validate)

__all__ = ["Episode", "EpisodeMeta", "Step", "Outcome", "validate", "SCHEMA_VERSION"]
