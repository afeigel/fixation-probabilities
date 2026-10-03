# Fluctuation theorems and exact multi-allele fixation probabilities

Emil A. Yuzbashyan, Alexander Feigel, and Alexandre V. Morozov

Code supplement for the article. The manuscript source is `article/fixation_probs_v2_combined.tex`.

The notebooks compute fixation probabilities under selection only: Wright–Fisher simulations, Langevin integration, and the multi-allele series expansion, including a beneficial mutation competing with other mutations (clonal interference).

Computed arrays are read and written in `data_fix_prob/`.

- `FixProbCode.ipynb` produces the numerical results: Wright–Fisher trajectories, Langevin (Euler–Maruyama) trajectories, the series expansion, and the clonal-interference scans, including the Kimura comparison.
- `FixProbPlots.ipynb` makes the article plots from those saved arrays: fixation-probability comparisons, the fluctuation-theorem constraint, and the clonal-interference heatmaps.

The plot notebook can be run on the arrays already stored in `data_fix_prob/` without repeating the simulations. Dependencies are NumPy, Numba, and Matplotlib.

Large files, including the symbolic partition cache `symbolic_partition_caches_maxk36.pkl`, are not stored in this repository. They will be deposited in a Zenodo repository.
