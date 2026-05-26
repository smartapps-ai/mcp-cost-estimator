import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .inference import CostEstimator, DATASET_SERVERS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

estimator: CostEstimator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global estimator
    logger.info("Loading cost estimator models...")
    estimator = CostEstimator()
    loaded = list(estimator.models.keys())
    if not loaded:
        logger.error("No models loaded — run train_model.py first")
    else:
        logger.info("Models loaded: %s", loaded)
    yield
    logger.info("Shutting down")


app = FastAPI(title="MCP Cost Estimator API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend origin in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

VALID_MODELS   = {"gpt-5.4"}
VALID_DATASETS = set(DATASET_SERVERS.keys())   # {"unitus", "umcu", "tpch"}


class EstimationRequest(BaseModel):
    question:  str
    gpt_model: str
    dataset:   str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v

    @field_validator("gpt_model")
    @classmethod
    def model_must_be_valid(cls, v: str) -> str:
        if v not in VALID_MODELS:
            raise ValueError(f"gpt_model must be one of {sorted(VALID_MODELS)}")
        return v

    @field_validator("dataset")
    @classmethod
    def dataset_must_be_valid(cls, v: str) -> str:
        if v not in VALID_DATASETS:
            raise ValueError(f"dataset must be one of {sorted(VALID_DATASETS)}")
        return v


@app.get("/health")
def health_check():
    loaded = list(estimator.models.keys()) if estimator else []
    return {"status": "ok", "models_loaded": loaded, "datasets": DATASET_SERVERS}


@app.get("/datasets")
def get_datasets():
    """Return the available dataset → server combinations."""
    return DATASET_SERVERS


@app.post("/estimate")
def get_estimation(request: EstimationRequest):
    if not estimator or not estimator.models:
        logger.error("Estimate requested but no models are loaded")
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run train_model.py and restart the server.",
        )

    logger.info(
        "Estimate request — question=%r gpt_model=%s dataset=%s",
        request.question[:80],
        request.gpt_model,
        request.dataset,
    )
    try:
        results = estimator.estimate(request.question, request.gpt_model, request.dataset)
        logger.info("Estimate complete — features=%s", results["inferred_features"])
        return results
    except Exception as exc:
        logger.exception("Unhandled error during estimation: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
