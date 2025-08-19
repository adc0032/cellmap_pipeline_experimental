import mlflow
import os
import json

def get_run_ids(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Get all runs in the experiment
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Filter to only child runs (those with a parent_run_id)
    child_runs_df = runs_df[runs_df['tags.mlflow.parentRunId'].notnull()]

    # Get the list of child run_ids
    child_run_ids = child_runs_df['run_id'].tolist()
    return child_run_ids

def get_run_info(experiment_name, run_id):
    # Get the MLflow metadata path for the run.
    run_path = get_run_path(experiment_name, run_id)
    run_results_path = f"{run_path}/artifacts/fairops/trial_results.json"
    
    # Load the run results metadata.
    with open (run_results_path, 'r') as f:
        run_results = json.load(f)

    # Transform the run parameters into a dictionary for ease of use.
    run_params = {p["key"]: p["value"] for p in run_results["params"]}

    return run_results, run_params

def get_run_path(experiment_name, run_id):
    # Get the experiment info (to find the artifact location)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # artifact_location might look like: 'file:///path/to/mlruns/12345'
    artifact_location = experiment.artifact_location

    # Convert file URI to local path (only if it's a local file store)
    if artifact_location.startswith("file://"):
        artifact_location = artifact_location[len("file://"):]

    # The run directory is just artifact_location/<run_id>
    run_dir = os.path.join(artifact_location, run_id)
    return run_dir

def get_run_uri(run_id):
    client = mlflow.tracking.MlflowClient()

    # Get the experiment ID
    upstream_run = client.get_run(run_id)
    experiment_id = upstream_run.info.experiment_id

    # Build the URL (edit if you're using a custom MLflow UI base)
    mlflow_ui_base = mlflow.get_tracking_uri().rstrip("/")  # e.g. http://localhost:5000
    run_url = f"{mlflow_ui_base}/#/experiments/{experiment_id}/runs/{run_id}"

    return run_url

def log_artifact_directory(dir_path, ignore_path=None):
    dir_path = os.path.abspath(dir_path)
    for root, _, files in os.walk(dir_path):
        for file in files:
            if ignore_path is not None and ignore_path in root:
                continue
            mlflow.log_artifact(os.path.join(root, file), "rocrate")

