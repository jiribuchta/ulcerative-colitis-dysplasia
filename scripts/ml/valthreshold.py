from kube_jobs import storage, submit_job


# MLflow run (Step 1) that holds the per-model val predictions:
# artifacts/valfold/<training_run_id>/val_predictions.parquet. Fill this in once
# the valfold job has finished.
valfold_run_id = "<valfold_run_id>"

# Step 2 is CPU-only (it just reads the saved predictions parquets and sweeps
# thresholds), so no GPU is requested.
submit_job(
    job_name="ulcerative-colitis-dysplasia-valthreshold",
    username="borisim",
    public=False,
    cpu=4,
    memory="8Gi",
    shm="2Gi",
    script=[
        "git clone -b feature/ml-cnn https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m ml +experiment=ml/valthreshold/virchow2 "
        "valthreshold.valfold_run_id='" + valfold_run_id + "'",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
