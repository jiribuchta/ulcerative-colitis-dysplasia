from kube_jobs import storage, submit_job


# Train step of the provenance demo: verifies the dataset against MLflow
# and fits a tiny model, with everything visible in MLflow.
#
# Prerequisites (once):
#   1. rationai-provenance library pushed to GitHub (uv dep below)
#   2. push this branch to the jiribuchta/ulcerative-colitis-dysplasia fork
#   3. create + register the demo dataset (a one-off job or local run):
#      uv run python -m provenance_demo +experiment=provenance_demo demo.step=create
#      uv run python -m provenance_demo +experiment=provenance_demo demo.step=register
submit_job(
    job_name="ulcerative-colitis-dysplasia-provenance-demo",
    username="jiribuchta",
    image="cerit.io/jiri_buchta/base-test-cuda:0.0.7",
    public=False,
    cpu=2,
    memory="4Gi",
    script=[
        "git clone https://github.com/jiribuchta/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "git checkout feature/provenance-demo",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow-jiribuchta.rationai-mlflow:5000/ PYTHONUNBUFFERED=1 && uv run python -u -m provenance_demo +experiment=provenance_demo demo.step=train",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
