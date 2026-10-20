DEMO DATA ONLY
==============

This folder exists to exercise the code and to let you check your install (python tests/smoke.py).
It is NOT a realistic use case, and scores obtained on it say nothing about how MultiPeptide will perform on real properties of your own proteins.

Why it is not representative:
  * its targets are by-products of structure prediction and database curation, not measurements you would want to predict:
    af_plddt (the AlphaFold confidence of each protein's model), is_catalytic (a flag derived from the Gene Ontology term GO:0003824)
    and go_terms (106 Gene Ontology molecular-function terms);
  * the interpro column (InterPro domain ids of each protein, from the curated manifest) is a real input, kept to exercise the domain-guided structure readout;
  * the 157 proteins are soluble monomers only, a small stride sample of the 634-protein curated set (train 123 / valid 17 / test 17).

Provenance: real UniProt sequences and real structures from the curated DPFunc monomer set in this repository's main branch
(data/curated_mf and its structure files). To keep the folder small, each structure file keeps only its header and the C-alpha atoms of the
chain used (the only atoms MultiPeptide reads); sequences, coordinates and B-factors are unchanged.
Structures are AlphaFold DB models (CC BY 4.0, Google DeepMind and EMBL-EBI) and experimental entries from the RCSB Protein Data Bank.
