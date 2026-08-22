from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-embeddings",
    username="jiribuchta",
    image="cerit.io/jiri_buchta/base-test-cuda:0.0.7",
    public=False,
    cpu=8,
    memory="32Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/jiribuchta/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "git checkout feature/embeddings",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow-jiribuchta.rationai-mlflow:5000/ PYTHONUNBUFFERED=1 MLFLOW_USER=jiribuchta && uv run python -u -m preprocessing.embeddings +dataset=tiled_filtered/level2_extent512",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
