from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-create-dataset",
    username="jiribuchta",
    public=False,
    cpu=8,
    memory="16Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "export MLFLOW_TRACKING_URI=http://mlflow-jiribuchta.rationai-mlflow:5000/ PYTHONUNBUFFERED=1 && uv run python -u -m preprocessing.create_dataset +dataset=raw",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
