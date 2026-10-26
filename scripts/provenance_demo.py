import time

from kube_jobs import storage, submit_job
from kube_jobs.cli.job import delete_job

JOB_NAME = "ulcerative-colitis-dysplasia-provenance-demo"

# rerunnable: remove the previous run's job (submit_job 409s on existing names)
delete_job(JOB_NAME, public=False)
time.sleep(5)  # foreground-propagated deletion needs a moment


# Train step of the provenance demo: verifies the dataset against MLflow
# and fits a tiny model, with everything visible in MLflow.
#
# Prerequisites (once):
#   1. push the rationai-provenance library to GitLab (uv dep below)
#   2. push this branch to the RationAI/ulcerative-colitis-dysplasia fork
#   3. create + register the demo dataset (a one-off job or local run):
#      uv run python -m provenance_demo +experiment=provenance_demo demo.step=create
#      uv run python -m provenance_demo +experiment=provenance_demo demo.step=register
submit_job(
    job_name=JOB_NAME,
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
        "export PYTHONUNBUFFERED=1",
        "uv run python -m provenance_demo +experiment=provenance_demo demo.step=create",
        "uv run python -m provenance_demo +experiment=provenance_demo demo.step=register",
        "uv run python -u -m provenance_demo +experiment=provenance_demo demo.step=train",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
