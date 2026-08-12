from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-predict",
    username="borisim",
    public=False,
    cpu=4,
    memory="32Gi",
    shm="16Gi",
    gpu="mig-1g.10gb",
    script=[
        "git clone -b feature/ml-cnn https://github.com/RationAI/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync --frozen",
        # Choose the embedding level at runtime: .../level1_extent224 or .../level2_extent224.
        # The checkpoint URI must stay single-quoted for Hydra (it contains a '=' in epoch=11).
        "uv run python -m ml +experiment=ml/predict/virchow2_test_preliminary +dataset=embeddings/level2_extent224 "
        "checkpoint=\\'mlflow-artifacts:/111/2f40594a222549feb7effc4e08447769/artifacts/checkpoints/epoch=11-step=4680/checkpoint.ckpt\\'",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
