# Experiments

One folder per experiment, numbered in the order they were started:

```
NN_short_name/
├── README.md          question, how to run, method, results, caveats; who runs it
├── <script>.py        takes --model (any key of jlens_open.MODELS) and --device
├── data/              small inputs that belong in git (prompt sets, ...)
├── results/           CSVs written by the script
└── figures/           PDF + PNG written by the script
```

Conventions:

- Scripts import shared code from `jlens_open` and write only inside their own folder. Code that a second
  experiment needs moves to `jlens_open/`.
- Results and figures are committed when they come from the committed code, so a PR shows code and numbers
  together. Files that are large or not reproducible from the repo (activations, fitted lenses, caches) go to
  `$JLENS_SHARED` on the cluster, never into git.
- Fix seeds; held-out splits are defined in the script, not by hand.
- Prompts that are read with a J-lens should be at least 16 tokens long: the lenses were fit without the first
  16 positions.
- Add a line for the new experiment to the table in the top-level README.
