"""Constants shared by the stimuli, the model runs and the analysis of experiment 10.

Experiment 10 asks which kind of surprise the model's inside reacts to: how unlikely a number was (surprisal), how
far the belief about the stream moved (Bayesian surprise), how far the belief now is from where it was before the
event (deviation), or how uncertain the reader has become. Five versions of every stream share the same random
draws and differ only from the event on (ARMS).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "10_unlikely_vs_update") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))
SKIP = 16  # the lenses were fit without the first 16 positions

LO, HI = 10, 990  # numbers are rounded and clipped to [10, 990], as in experiment 09 (Sarfati et al. 2026)
N_SEQ = 20
LEN = 600         # numbers per stream
TAU = 400         # index of the first number after the event (numbers 0..399 are the same in every arm)
MU0, SIGMA0 = 500, 60
DELTA = 300       # size of the jump, the outlier and the drift (5 sd); up for even k, down for odd k
DRIFT = 100       # the drift reaches the new level after this many numbers (3 per number: 0.05 sd)
NOISE = 3         # the noise arm multiplies the sd by this from TAU on (60 -> 180), same mean
SEED = 2000       # stream k uses the standard normal draws of default_rng(SEED + k), in every arm
ARMS = ("control", "jump", "outlier", "drift", "noise")
# Cut points for the cue readouts: e = t - TAU is the last number read (e = -1: before the event, all arms equal)
CUTS = (-1, 0, 1, 2, 4, 8, 32, 128)
CUES = {
    "wait": ",\n\nWait,",
    "last": ",\n\nThe last number is",
}

# Word sets read at every position (log-sum-exp of the log-softmax over their single-token " word" forms). The
# surprise, change and normal sets are experiment 09's cue sets; magnitude and the decades are its word sets
# (wordsets.py); control is its control list.
SETS = {
    "surprise": ["surprising", "unexpected", "unusual", "strange", "odd", "weird", "outlier", "anomaly", "surprise",
                 "sudden", "suddenly", "wrong", "error", "interesting", "wait"],
    "change": ["change", "changed", "changing", "different", "new", "shift", "jump", "switch", "increased",
               "increasing", "higher", "rising", "bigger", "now", "moved"],
    "normal": ["normal", "typical", "expected", "usual", "consistent", "similar", "same", "fine", "ok", "random",
               "stable", "average", "around", "about", "between"],
    "magnitude": ["trillion", "billion", "billions", "million", "millions", "thousand", "thousands", "infinite",
                  "huge", "hundreds"],
    "wide": ["wide", "spread", "scattered", "varied", "variable", "volatile", "noisy", "vary", "fluctuate", "uncertain"],
    "narrow": ["narrow", "tight", "stable", "steady", "constant", "consistent", "close", "similar", "same", "precise"],
    "high": ["high", "higher", "large", "larger", "big", "bigger", "above", "more", "increase", "up", "many"],
    "low": ["low", "lower", "small", "smaller", "below", "less", "decrease", "down", "little", "few"],
    **{d: [d, d.capitalize()] for d in ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty")},
    "control": ["table", "river", "green", "letter", "music", "paper", "window", "garden", "chair", "yellow",
                "bottle", "mountain", "pencil", "coffee", "island", "horse", "cloud", "kitchen", "forest", "shirt"],
}


def band(n_blocks):
    """The middle third of the layers (as in experiments 08 and 09; Llama-3.1-8B: 10-21)."""
    return list(range(n_blocks // 3, (2 * n_blocks) // 3 + 1))
