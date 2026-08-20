from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-embeddings",
    username="jiribuchta",
    public=False,
    cpu=8,
    memory="16Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/jiribuchta/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "git checkout feat/embeddings",
        "uv sync --frozen",
        "uv run python -m preprocessing.embeddings +preprocessing=embeddings",
    ],
    storage=[storage.secure.DATA],
)
