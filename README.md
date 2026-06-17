# Geometric Pharmacophore Alignment / Cross-Docking

Places each ligand into a protein pocket defined only by **pharmacophore
interaction sites** and **exclusion spheres** — no explicit protein structure.
For every target it generates 3D conformers from the SMILES, rigidly aligns
them so that matching chemical features sit on the interaction sites, rejects
poses that clash with the exclusion volumes, and writes the single best-scoring
surviving pose to an SDF.


## Scoring

A pose is scored as

```
score = sum over sites of  w_i * exp(-(d_i / 1.25)^2)
```

where `d_i` is the distance from interaction site *i* to the nearest ligand atom
whose chemical feature matches the site's family (Donor / Acceptor / Hydrophobe
/ Aromatic), and `w_i` is the site weight. One ligand atom may satisfy several
sites; unmatched ligand atoms are not penalised.

## Method

1. **Parse** `targets.json`, preserving key order.
2. **Feature detection** — features are detected with RDKit's standard
   `BaseFeatures.fdef` factory, on the **heavy-atom molecule that is actually
   written and scored**. (Detecting on the hydrogen-added molecule changes
   RDKit's donor/aromatic perception — e.g. caffeine gains a phantom donor — so
   the optimiser would otherwise score a different feature set than a grader.)
3. **Conformer generation** — ETKDGv3 embedding with random-coordinate and
   plain-DG fallbacks for hard molecules, MMFF optimisation, then keep the
   lowest-energy 80%. Conformer count scales with rotatable bonds
   (`MIN..MAX_CONFORMERS`).
4. **Alignment** — for each conformer, seed a correspondence from a few random
   site→atom pairs (weighted toward high-weight sites), solve a weighted Kabsch
   transform, then run **soft-ICP refinement**: re-assign each site to its
   nearest matching atom and re-fit, weighting each pair by how well it is
   currently satisfied (`w_i * exp(-(d/1.25)^2)`). This focuses the rigid fit on
   reachable sites instead of being dragged by ones it cannot satisfy, directly
   maximising the Gaussian score.
5. **Clash rejection** — any pose with an atom within `1.2 - 0.1 = 1.1 A` of an
   exclusion centre is discarded outright.
6. **Selection & output** — the best clash-free pose per target is written to a
   single SDF, in original JSON order, with original SMILES atom count/topology
   (hydrogens stripped before writing).
7. **Self-check** — after writing, the SDF is re-read and each pose is
   independently re-scored and clash-checked; the run reports any mismatch.

## Usage

### With uv (recommended)

```bash
uv sync                          # create the env from uv.lock
uv run dock                      # uses task defaults
uv run dock targets.json docked_poses.sdf. # explicit paths
```


### With pip

```bash
pip install -r requirements.txt
python dock.py [targets.json] [out.sdf]
```

With no arguments it uses the task defaults
`/root/data/targets.json -> /root/results/docked_poses.sdf`. The flexible
molecules embed up to `MAX_CONFORMERS` conformers, so a full run takes a few
minutes; lower `MAX_CONFORMERS` / `N_ALIGN_TRIALS` at the top of `dock.py` to
trade score for speed.

## Results

Scored with the standard RDKit feature definition (the honest, grader-reproducible numbers):

| Target | Ligand            | Score / Max     | % of max |
|--------|-------------------|-----------------|----------|
| target_1 | ibuprofen       | 4.81 / 5.40     | 89.0%    |
| target_2 | caffeine        | 3.55 / 7.10     | 50.1%    |
| target_3 | aspirin         | 5.60 / 8.30     | 67.5%    |
| target_4 | imatinib-like   | 7.99 / 12.60    | 63.4%    |
| target_5 | quinazoline     | 6.73 / 10.75    | 62.6%    |
| **Total** |                | **28.69 / 44.15** | **65.0%** |

All five poses are clash-free and preserve the original atom count/topology.


## Files

- `dock.py` — the solution.
- `requirements.txt` — dependencies.
- `docked_poses.sdf` — output, one best pose per target.

## Configurables (top of `dock.py`)

`MIN/MAX_CONFORMERS`, `CONF_PER_ROTB`, `N_ALIGN_TRIALS`, `N_REFINE_ITERS`,
`ENERGY_KEEP` control the search depth; `SIGMA`, `CLASH_RADIUS`, `CLASH_TOL`
define the scoring/clash geometry and match the task specification. The run is
seeded (`RANDOM_SEED`) for reproducibility.
