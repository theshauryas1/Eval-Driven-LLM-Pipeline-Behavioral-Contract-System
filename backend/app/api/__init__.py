from fastapi import APIRouter

from app.api.traces import router as traces_router
from app.api.contracts import router as contracts_router
from app.api.results import router as results_router

api_router = APIRouter()
api_router.include_router(traces_router)
api_router.include_router(contracts_router)
api_router.include_router(results_router)
