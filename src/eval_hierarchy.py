import os
import json

import mlflow
from mlops_helper import get_run_ids, get_run_uri, log_artifact_directory

from cellmaps_hierarchyeval.runner import CellmapshierarchyevalRunner
from fairops.mlops.autolog import LoggerFactory


mlflow.set_experiment("hierarchyeval")
ml_logger = LoggerFactory.get_logger("mlflow")

configs_file_path = "./configs/eval_hierarchy_configs.json"

with open (configs_file_path, 'r') as f:
    configs = json.load(f)
config = configs[0]

with mlflow.start_run() as parent_run:
    mlflow.set_tag("pipeline_step", "cellmaps_hierarchyeval_parent")

    run_ids = get_run_ids("hierarchy")
    mlflow.log_param("n_trials", len(run_ids))

    for run_id in run_ids:
        with mlflow.start_run(nested=True) as child_run:
            hiergen_dir = f"data/hierarchy/generator/{run_id}"
            hierarchy_run_uri = get_run_uri(run_id)

            mlflow.set_tag("pipeline_step", "cellmaps_hierarchyeval")
            mlflow.log_param("hierarchy_run_id", run_id)
            mlflow.log_param("hierarchy_run_uri", hierarchy_run_uri)

            hiereval_dir = f"data/hierarchy/eval/{child_run.info.run_id}"
            
            eval_hierarchy = CellmapshierarchyevalRunner(
                outdir=hiereval_dir,
                hierarchy_dir=hiergen_dir,
                max_fdr=config["max_fdr"],
                min_jaccard_index=config["min_jaccard_index"],
                min_comp_size=config["min_comp_size"],
                corum=config["corum"],
                go_cc=config["go_cc"],
                hpa=config["hpa"],
                log_fairops=True
            )

            eval_hierarchy.run()

            hierarchyeval_rocrate_path = log_artifact_directory(hiereval_dir)
            
            ml_logger.export_logs_as_artifact()
            mlflow.end_run()
    
    ml_logger.export_logs_as_artifact()
    mlflow.end_run()
