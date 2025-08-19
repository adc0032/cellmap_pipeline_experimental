import os
import csv

from mlops_helper import get_run_ids, get_run_info

run_ids = get_run_ids("hierarchyeval")

eval_results_header = [
    "coembed_run_id",
    "coembed_algorithm",
    "hierarchy_run_id",
    "hierarchy_algorithm",
    "hierarchy_k",
    "hierarchy_maxres",
    "hierarchy_containment_threshold",
    "hierarchy_jaccard_threshold",
    "hierarchy_min_diff",
    "hierarchy_min_system_size",
    "hierarchy_ppi_cutoffs",
    "hierarchy_parent_ppi_cutoffs",
    "hierarchy_bootstrap_edges",
    "hierarchy_eval_run_id",
    "hierarchy_mean_corum_jaccard",
    "hierarchy_mean_go_cc_jaccard",
    "hierarchy_mean_hpa_jaccard",
]
eval_results_rows = [eval_results_header]
for run_id in run_ids:
    # Retrieve the results for hierarchy evaluation, hierarchy generation, and coembedding.
    eval_results, eval_params = get_run_info(
        "hierarchyeval", 
        run_id
    )
    hierarchy_results, hierarchy_params = get_run_info(
        "hierarchy", 
        eval_params["hierarchy_run_id"]
    )
    coembed_results, coembed_params = get_run_info(
        "coembedding", 
        hierarchy_params["coembed_run_id"]
    )

    # Add evaluation results to the csv array.
    eval_results_rows.append([
        coembed_results["run_id"],
        coembed_params["algorithm"], 
        hierarchy_results["run_id"],
        hierarchy_params['algorithm'],
        hierarchy_params["k"],
        hierarchy_params["maxres"],
        hierarchy_params["containment_threshold"],
        hierarchy_params["jaccard_threshold"],
        hierarchy_params["min_diff"],
        hierarchy_params["min_system_size"],
        hierarchy_params["ppi_cutoffs"],
        hierarchy_params["parent_ppi_cutoff"],
        hierarchy_params["bootstrap_edges"],
        eval_results['run_id'],
        eval_results['last_metrics']['hierarchy_mean_corum_jaccard'],
        eval_results['last_metrics']['hierarchy_mean_go_cc_jaccard'],
        eval_results['last_metrics']['hierarchy_mean_hpa_jaccard'],
    ])

# Create and write to the CSV file.
with open("data/hierarchy_eval_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(eval_results_rows)
