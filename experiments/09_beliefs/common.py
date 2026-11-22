"""Constants shared by the stimuli, the model runs and the analysis of experiment 09."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# Raw model outputs (too large for git): the group's storage on the cluster, else raw/ here (git-ignored)
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "09_beliefs") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))
SKIP = 16  # the lenses were fit without the first 16 positions

# The stimuli of Sarfati et al. (2026), "The Shape of Beliefs" (arXiv:2602.02315): generate_sequences.py and
# scripts/generate_all.sh of github.com/raphael-goodfire/shape-of-beliefs. Integers drawn from N(mu, sigma), rounded,
# clipped to [10, 990] and written "533,460,689,...," (no spaces, a comma after the last number). Every condition
# is drawn with the same seed, so sequence k of every condition uses the same standard normal draws: conditions
# differ only by (mu, sigma).
SEED = 18
LO, HI = 10, 990
DELIM = ","
N_SEQ = 10
LEN = 1000
SIGMA0, MU0 = 100, 500
MUS = (300, 350, 400, 450, 500, 550, 600, 650, 700)          # at sigma = 100
SIGMAS = (10, 20, 30, 50, 80, 100, 120, 150, 200)            # at mu = 500
CONDITIONS = [(m, SIGMA0) for m in MUS] + [(MU0, s) for s in SIGMAS if s != SIGMA0]  # 17 (500, 100 is shared)
SWITCH = ((300, 100), (700, 100))  # the paper's switch (its Fig. 2): 1000 numbers of each, concatenated


def cond_name(mu, sigma):
    return f"m{mu}_s{sigma:03d}"  # the paper's dataset names: gaussian_m{mu}_s{sigma:03d}_l1000_n10


# Token positions (the paper's names). With BOS first, number t (0-indexed) is token 2t + 1 and the comma after it
# token 2t + 2:
#   num2com  the token of number t (predicts the comma)
#   com2num  the comma after number t (predicts number t + 1): the paper's main position, where the belief is read
POS = ("com2num", "num2com")

# Bins of the number index t for the cell means (stationary sequences). t < 8 is before token 16, outside the
# positions the lenses were fit on.
T_BINS = (0, 8, 32, 128, 512, 1000)
# Bins of event time e = t - 1000 for the switch sequences (e = 0 is the first number drawn from N(700, 100))
E_BINS = (-1000, -500, -100, -10, 0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1000)


def bin_labels(edges):
    return [f"[{a},{b})" for a, b in zip(edges[:-1], edges[1:])]


# Words whose lens readout is recorded at every position (each as " word"; words that are not one token are dropped
# at run time). A group's score is the mean z-score (over the vocabulary) of its words minus that of "control".
LEXICON = {
    "stats": ["random", "average", "mean", "normal", "distribution", "variance", "deviation", "standard", "sample",
              "noise", "gaussian", "statistics", "data", "numbers", "values", "range", "probability", "estimate"],
    "change": ["change", "changed", "shift", "shifted", "jump", "jumped", "switch", "new", "different", "sudden",
               "suddenly", "now"],
    "surprise": ["wait", "actually", "unexpected", "surprise", "surprising", "strange", "odd", "weird", "unusual",
                 "hmm", "but", "however"],
    "high": ["high", "higher", "large", "larger", "big", "bigger", "above", "more", "increase", "up", "many"],
    "low": ["low", "lower", "small", "smaller", "below", "less", "decrease", "down", "little", "few"],
    "wide": ["wide", "spread", "scattered", "varied", "variable", "volatile", "noisy", "vary", "fluctuate", "uncertain"],
    "narrow": ["narrow", "tight", "stable", "steady", "constant", "consistent", "close", "similar", "same", "precise"],
    "control": ["table", "river", "green", "letter", "music", "paper", "window", "garden", "chair", "yellow",
                "bottle", "mountain", "pencil", "coffee", "island", "horse", "cloud", "kitchen", "forest", "shirt"],
}
GROUPS = tuple(LEXICON)


def band(n_blocks):
    """The middle third of the layers (as in experiment 08; Llama-3.1-8B: 10-21)."""
    return list(range(n_blocks // 3, (2 * n_blocks) // 3 + 1))


def thirds(n_blocks):
    """Early, middle and late thirds of the layers, for the word tables."""
    b = band(n_blocks)
    return {"early": list(range(0, b[0])), "middle": b, "late": list(range(b[-1] + 1, n_blocks))}
