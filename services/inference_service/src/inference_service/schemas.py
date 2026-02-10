from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120 , description="Age of the customer")
    city: str = Field(..., description="City where the customer lives")
    income: float = Field(..., ge=0, description="Annual income of the customer")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float = Field(..., ge=0.0, le=1.0 , description="Probability of the positive class")
    model_version: str = Field(..., description="Version of the model used for prediction")