"""Layer 2 — data standard & capture (doc §5).

The open standard for a captured intervention: a time-synchronized log of device
kinematics + the paired Layer-1 observation + outcome (`schema.Episode`), plus the
machinery to produce it from the simulator (`capture`), iterate a corpus
(`replay`), and close the §3.6 calibration loop on it (`calibrate`). The real
patient corpus is the proprietary moat (§327) and stays private behind the same
`Episode` seam — never in this repo (firewall: provenance == "procedural").
"""

from lumen.data.bundle import CaseBundle, validate_case_bundle
from lumen.data.calibrate import calibrate_from_episode, joint_probe_episode, probe_episode
from lumen.data.capture import EpisodeRecorder, rollout_episode
from lumen.data.index import (device_keypoint_mask_distances, device_keypoint_mask_errors,
                              iter_index_records, iter_step_records, load_step_record,
                              resolve_record_paths, summarize_index)
from lumen.data.materialize import materialize_index_batch
from lumen.data.metrics import compute_clinical_metrics
from lumen.data.replay import EpisodeDataset, annotation_coverage, replay, summarize
from lumen.data.schema import (SCHEMA_VERSION, Episode, EpisodeMeta, Outcome, Step,
                               validate)

__all__ = ["Episode", "EpisodeMeta", "Step", "Outcome", "validate", "SCHEMA_VERSION",
           "CaseBundle", "validate_case_bundle",
           "compute_clinical_metrics",
           "EpisodeRecorder", "rollout_episode",
           "EpisodeDataset", "replay", "summarize", "annotation_coverage",
           "iter_step_records", "iter_index_records", "load_step_record",
           "resolve_record_paths", "summarize_index", "device_keypoint_mask_errors",
           "device_keypoint_mask_distances",
           "materialize_index_batch",
           "probe_episode", "joint_probe_episode", "calibrate_from_episode"]
