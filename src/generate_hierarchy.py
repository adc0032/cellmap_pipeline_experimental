import os
import json
import itertools


import mlflow
from mlops_helper import get_run_ids, get_run_uri, log_artifact_directory

from cellmaps_generate_hierarchy.hcx import HCXFromCDAPSCXHierarchy
from cellmaps_generate_hierarchy.hierarchy import CDAPSHiDeFHierarchyGenerator
from cellmaps_generate_hierarchy.maturehierarchy import HiDeFHierarchyRefiner
from cellmaps_generate_hierarchy.ppi import CosineSimilarityPPIGenerator
from cellmaps_generate_hierarchy.runner import CellmapsGenerateHierarchy
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("hierarchy")
ml_logger = LoggerFactory.get_logger("mlflow")

# Define lists of possible values for each key
algorithms = ["leiden", "louvain", "walktrap"]
ks = [10]
maxres = [80]
containment_thresholds = [0.8]
jaccard_thresholds = [0.8, 0.85, 0.9, 0.95]
min_diffs = [1]
min_system_sizes = [2, 3, 4]
ppi_cutoffs = [[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]]
parent_ppi_cutoffs = [0.1]
bootstrap_edges = [0]

# Generate Cartesian product of all parameter values
combinations = itertools.product(
    algorithms,
    ks,
    maxres,
    containment_thresholds,
    jaccard_thresholds,
    min_diffs,
    min_system_sizes,
    ppi_cutoffs,
    parent_ppi_cutoffs,
    bootstrap_edges
)

# Build list of dictionaries
configs = [
    {
        "algorithm": a,
        "k": k,
        "maxres": m,
        "containment_threshold": ct,
        "jaccard_threshold": jt,
        "min_diff": md,
        "min_system_size": ms,
        "ppi_cutoffs": pc,
        "parent_ppi_cutoff": ppc,
        "bootstrap_edges": be
    }
    for (a, k, m, ct, jt, md, ms, pc, ppc, be) in combinations
]

run_ids = get_run_ids("coembedding")

with mlflow.start_run() as parent_run:
    mlflow.set_tag("pipeline_step", "cellmaps_generate_hierarchy_parent")
    mlflow.log_param("n_trials", len(configs) * len(run_ids))

    for run_id, config in itertools.product(run_ids, configs):
        with mlflow.start_run(nested=True) as child_run:
            coembed_dir = f"data/embedding/coembed/{run_id}"
            config["coembed_run_id"] = run_id
            config["coembed_run_uri"] = get_run_uri(run_id)
            
            mlflow.set_tag("pipeline_step", "cellmaps_generate_hierarchy")
            mlflow.log_params(config)

            hiergen_dir = f"data/hierarchy/generator/{child_run.info.run_id}"

            ppigen = CosineSimilarityPPIGenerator(
                embeddingdirs=[coembed_dir],
                cutoffs=config["ppi_cutoffs"]
            )
            
            refiner = HiDeFHierarchyRefiner(
                ci_thre=config["containment_threshold"],
                ji_thre=config["jaccard_threshold"],
                min_term_size=config["min_system_size"],
                min_diff=config["min_diff"]
            )
            converter = HCXFromCDAPSCXHierarchy()
            hiergen = CDAPSHiDeFHierarchyGenerator(
                refiner=refiner,
                hcxconverter=converter,
                hierarchy_parent_cutoff=config["parent_ppi_cutoff"],
                bootstrap_edges=config["bootstrap_edges"]
            )

            generate_hierarchy = CellmapsGenerateHierarchy(
                outdir=hiergen_dir,
                inputdirs=coembed_dir,
                ppigen=ppigen,
                hiergen=hiergen,
                algorithm=config["algorithm"],
                maxres=config["maxres"],
                k=config["k"]
            )

            generate_hierarchy.run()

            hierarchy_rocrate_path = log_artifact_directory(hiergen_dir)

            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()
