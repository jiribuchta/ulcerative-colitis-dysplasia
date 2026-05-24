from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-mean-std",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",  # approximately 4GiB per process
    public=False,
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run -m preprocessing.mean_std +dataset=...",
    ],
    storage=[storage.secure.DATA],
)
