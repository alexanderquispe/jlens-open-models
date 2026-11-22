"""Constants shared by the stimuli and the analysis of experiment 12 (how the old belief dies).

Two kinds of stream, all with noise sd SIGMA (as in experiment 09):
  jumps   the level jumps once, at number TAU, by DELTAS (up or down, centred on 500), plus a flat control
  ramps   the level slides from one value to the other and back (up-down and down-up), fast or slow, for the
          hysteresis test: does the switch point depend on the direction of approach, beyond the lag any
          reader with memory has?
  history one jump after a stable stretch of varying length (HIST_TAUS): does the old belief die more slowly the
          longer it has held? A running mean predicts a switch time proportional to the history length, a moving
          average and the change-point observer an (almost) constant one.
The model runs reuse experiment 10's run_streams.py (same recorded arrays).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "12_belief_survival") if os.environ.get("JLENS_SHARED")
       else os.path.join(HERE, "raw"))
EXP09_RAW = (os.path.join(os.environ["JLENS_SHARED"], "experiments", "09_beliefs") if os.environ.get("JLENS_SHARED")
             else None)

LO, HI = 10, 990
SIGMA = 100
N_SEQ = 20
# Jumps: level 500 - s DELTA / 2 before TAU and 500 + s DELTA / 2 from TAU on (s = +1 up, -1 down)
JUMP_LEN, TAU = 800, 500
DELTAS = (100, 200, 400)
JUMP_SEED = 3000
# Ramps: flat at A for PRE numbers, slide to B over R numbers, stay PLATEAU numbers, slide back over R, flat POST
LOW, HIGH = 300, 700
PRE, PLATEAU, POST = 300, 200, 200
RAMPS = {"fast": 50, "slow": 200}
RAMP_SEED = 4000
# History length: a jump of HIST_DELTAS (up / down, centred on 500) after HIST_TAUS numbers at the old level, then
# HIST_POST numbers at the new level; the draws just before and after the jump are the same for every history length
HIST_TAUS = (25, 50, 100, 200, 400, 800)
HIST_DELTAS = (200, 400)
HIST_POST = 200
HIST_SEED = 5000
