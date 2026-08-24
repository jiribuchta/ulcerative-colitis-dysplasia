from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-create-dataset",
    username="jiribuchta",
    image="cerit.io/jiri_buchta/base-test-cuda:0.0.7",
    public=False,
    cpu=4,
    memory="16Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/jiribuchta/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow-jiribuchta.rationai-mlflow:5000/ PYTHONUNBUFFERED=1 MLFLOW_USER=jiribuchta && uv run python -u -m preprocessing.create_dataset +dataset=raw",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
