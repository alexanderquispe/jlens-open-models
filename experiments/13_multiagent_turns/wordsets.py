"""Word sets read in the lens, fixed on 2026-11-16 before any lens output on agent prompts was looked at.

A word counts through the first token of "word" and of " word" (LensModel.first_token), as in experiments 09 to 11.
"""
SETS = {
    # what the agent is about to do
    "act": ["scale", "cancel", "pause", "stop", "cut", "reduce", "kill", "launch", "start", "increase", "boost",
            "allocate", "shift", "move"],
    "wait": ["wait", "hold", "later", "idle", "sleep", "patience", "monitor", "watch", "check", "nothing", "pending"],
    "ask": ["ask", "permission", "consent", "approve", "agree", "confirm", "request", "message", "tell", "notify",
            "negotiate", "coordinate"],
    # what it has registered about the situation
    "scarce": ["budget", "limit", "cap", "short", "shortage", "scarce", "insufficient", "exceed", "overrun", "tight",
               "deficit", "constraint"],
    "fair": ["fair", "share", "equal", "split", "everyone", "priority", "urgent", "important", "deadline", "tradeoff"],
    "people": ["Nadia", "Viktor", "Rosa", "Omar"],
    # experiment 11's sets, for comparison with its results
    "surprise": ["surprising", "unexpected", "unusual", "strange", "odd", "weird", "outlier", "anomaly", "surprise",
                 "sudden", "suddenly", "wrong", "error", "interesting", "wait"],
    "control": ["table", "river", "green", "letter", "music", "paper", "window", "garden", "chair", "yellow", "bottle",
                "mountain", "pencil", "coffee", "island", "horse", "cloud", "kitchen", "forest", "shirt"],
}
# The tools of the API key environment, by the word that follows "labctl_" (or starts the name)
TOOLS = {"labctl_scale": "scale", "labctl_cancel": "cancel", "labctl_submit": "submit", "labctl_status": "status",
         "labctl_dashboard": "dashboard", "labctl_wait": "wait", "labctl_update_estimate": "update",
         "labctl_set_account": "set", "SendMessage": "Send", "find_agents": "find", "read": "read", "write": "write",
         "edit": "edit", "bash": "bash"}
PRINCIPALS = {"a": "Nadia", "b": "Viktor", "c": "Omar", "d": "Rosa"}  # job j-a01 belongs to Nadia, and so on (world summary)


def words():
    """Every word whose lens readout is kept."""
    out = []
    for group in list(SETS.values()) + [list(TOOLS.values())]:
        out += [w for w in group if w not in out]
    return out
