from typing import List
from pydantic import BaseModel, ConfigDict, field_validator

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    model_config = ConfigDict(extra="forbid")

    @field_validator("Latitude")
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Invalid value for Latitude")
        return v

    @field_validator("Longitude")
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Invalid value for Longitude")
        return v


class BatchHousingInput(BaseModel):
    houses: List[HousingInput]

    model_config = ConfigDict(extra="forbid")

    # Method to vectorize the housing input for batch prediction
    def vectorize(self) -> List[List[float]]:
        return [
            [
                house.MedInc,
                house.HouseAge,
                house.AveRooms,
                house.AveBedrms,
                house.Population,
                house.AveOccup,
                house.Latitude,
                house.Longitude,
            ]
            for house in self.houses
        ]


class PredictionOutput(BaseModel):
    prediction: float


class BatchPredictionOutput(BaseModel):
    predictions: List[float]


class HealthResponse(BaseModel):
    status: str


class HelloResponse(BaseModel):
    message: str
