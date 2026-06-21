# TODO - Optimization plan

## Step 0 - Repo understanding
- [x] Reviewed main pipeline entrypoints (MPI + orchestrator)
- [x] Identified likely hotspots: RDF kernel, snapshot spatial work, serialization/CSV export

## Step 1 - Optimize RDF kernel (`src/rdf.py`)
- [x] Rewrite RDF accumulation to reduce memory + improve speed
- [x] Ensure output structure remains backward compatible (dict of atom_id -> g(r) array)
- [x] Add sanity assertions / validation helpers if needed




## Step 2 - Remove redundant spatial indexing (`src/snapshot_processor.py`)
- [x] Avoid building KDTree if RDF kernel already handles spatial querying (or remove duplicated setup)
- [x] Ensure Voronoi and neighbor queries still work
- [x] Keep output keys identical



## Step 3 - Serialization/merge improvements
- [ ] Update `pipeline_orchestrator.py` to write partials to disk to reduce pickling overhead
- [ ] Update `pipeline_mpi.py` partials format (prefer compact numeric formats)

## Step 4 - Optimize CSV export
- [ ] Reduce per-row Python overhead in `csv_export`
- [ ] Keep CSV schema stable

## Step 5 - Validation
- [ ] Run pipeline on small subset and compare peak outputs
- [ ] Measure runtime improvements using new timing logs

