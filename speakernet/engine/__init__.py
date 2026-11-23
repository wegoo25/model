"""Training / evaluation engine."""
from speakernet.engine.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from speakernet.engine.evaluation import compute_metrics, evaluate_from_list, extract_embeddings, score_trials
from speakernet.engine.speaker_net import SpeakerNet
from speakernet.engine.trainer import Trainer

__all__ = [
    "SpeakerNet", "Trainer", "save_checkpoint", "load_checkpoint", "latest_checkpoint",
    "evaluate_from_list", "compute_metrics", "extract_embeddings", "score_trials",
]
