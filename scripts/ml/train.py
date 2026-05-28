from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-train-...",
    username="...",
    public=False,
    cpu=16,
    memory="64Gi",
    shm="64Gi",
    gpu="mig-1g.10gb",
    script=[
        "git clone https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        "uv run python -m ml +dataset=... +experiment=ml/train/... val_fold=0,1,2,3,4 --multirun",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
