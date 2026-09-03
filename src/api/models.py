"""
models.py  —  Pydantic request / response schemas for the FastAPI server.
"""
from pydantic import BaseModel, Field
from typing import Optional


class VerifyRequest(BaseModel):
    query:        str = Field(..., min_length=1,
                              description="The original user question.")
    llm_response: str = Field(..., min_length=1,
                              description="The LLM-generated answer to verify.")
    use_nli:      bool = Field(True,
                               description="Use NLI model in SCA. "
                                           "Set False for faster response.")


class Breakdown(BaseModel):
    sds:       float
    fva_score: float
    sca_score: float
    sds_risk:  float
    fva_risk:  float
    sca_risk:  float


class SVDAnalysis(BaseModel):
    sds:          float
    sigma_values: list[float]
    sigma_min:    float
    sigma_max:    float
    sigma_mean:   float
    k_used:       int
    context_shape: list[int]
    output_shape:  list[int]


class VerifyResponse(BaseModel):
    verdict:          str
    final_risk_score: float
    action:           str
    fva_status:       str
    breakdown:        Breakdown
    svd_analysis:     Optional[SVDAnalysis] = None
    query:            str
    llm_response:     str


class HealthResponse(BaseModel):
    status:  str
    message: str
