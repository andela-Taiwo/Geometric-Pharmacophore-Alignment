# Geometric Pharmacophore Alignment / Cross-Docking

Places each ligand into a protein pocket defined only by **pharmacophore
interaction sites** and **exclusion spheres** — no explicit protein structure.
For every target it generates 3D conformers from the SMILES, rigidly aligns
them so that matching chemical features sit on the interaction sites, rejects
poses that clash with the exclusion volumes, and writes the single best-scoring
surviving pose to an SDF.
