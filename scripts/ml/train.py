from kube_jobs import storage, submit_job


submit_job(
    job_name="ulcerative-colitis-dysplasia-train-new",
    username="jiribuchta",
    image="cerit.io/jiri_buchta/base-test-cuda:0.0.7",
    public=False,
    cpu=16,
    memory="64Gi",
    shm="64Gi",
    gpu="A40",
    script=[
        "git clone -b feature/ml-cnn https://github.com/jiribuchta/ulcerative-colitis-dysplasia.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow-jiribuchta.rationai-mlflow:5000/ PYTHONUNBUFFERED=1 MLFLOW_USER=jiribuchta && uv run python -m ml '+experiment=ml/train/virchow2_l2_epi0_allslides' val_fold=0,1,2,3,4 --multirun",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
