"""Constants shared by the stimuli, the model run and the analysis of experiment 08."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# Raw model outputs (too large for git): the group's storage on the cluster, else raw/ here (git-ignored)
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "08_surprise") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))
SKIP = 16  # the lenses were fit without the first 16 positions

# Part A: each sequence is PREFIX followed by the numbers joined by SEP
PREFIX = "The following are readings from a sensor:\n"
SEP = ", "
T = 30  # numbers per sequence
ARMS = ("control", "jump", "outlier")

# Part B: the four versions of each solution
TWINS = ("clean", "near", "far", "swap")

# Positions read around each number x_t of Part A:
#   num   the token of x_t itself (differs across arms after the change, so it mixes token identity with context)
#   sep   the token right after x_t (the comma): the same token in every arm, so arm differences are context only
#   pred  the token right before x_{t+1}, where the model predicts the next number (Llama: the space)
SLOTS = ("num", "sep", "pred")

# Words whose J-lens readout is recorded (each as " word"; words that are not one token are dropped at run time).
# "control" is a set of neutral words, so that a group's score is its readout minus the control readout.
LEXICON = {
    "surprise": ["wait", "actually", "suddenly", "sudden", "unexpected", "unexpectedly", "surprise", "surprising",
                 "surprised", "strange", "odd", "weird", "wow", "whoa", "hmm", "unusual", "but", "however"],
    "error": ["wrong", "error", "mistake", "incorrect", "oops", "mistaken", "false", "typo", "incorrectly", "fix"],
    "change": ["change", "changed", "jump", "jumped", "shift", "shifted", "spike", "anomaly", "outlier", "different"],
    "up": ["higher", "increase", "increased", "rise", "rose", "up", "above", "larger", "bigger", "more"],
    "down": ["lower", "decrease", "decreased", "drop", "dropped", "fell", "down", "below", "smaller", "less"],
    "control": ["table", "river", "green", "letter", "music", "paper", "window", "garden", "chair", "yellow",
                "bottle", "mountain", "pencil", "coffee", "island", "horse", "cloud", "kitchen", "forest", "shirt"],
}
GROUPS = tuple(LEXICON)


def decomp_layers(n_blocks):
    """Eight layers spread over the depth where the J-space is decomposed (Llama-3.1-8B: 4 8 12 16 19 23 27 31)."""
    return sorted({round(i * (n_blocks - 1) / 8) for i in range(1, 9)})


def probe_layers(n_blocks):
    """Layers whose hidden states are saved for belief probes (Llama-3.1-8B: 8 16 20 24)."""
    return sorted({n_blocks // 4, n_blocks // 2, (5 * n_blocks) // 8, (3 * n_blocks) // 4})


def band(n_blocks):
    """The middle third of the layers, the pre-registered band for the headline numbers."""
    return list(range(n_blocks // 3, (2 * n_blocks) // 3 + 1))
