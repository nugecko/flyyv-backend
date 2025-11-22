from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later to your Base44 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Flyvo backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Search request model ----
class SearchRequest(BaseModel):
    origin: str
    destination: str
    earliestDeparture: str
    latestDeparture: str
    minStayDays: int
    maxStayDays: int
    maxPrice: float
    cabin: str
    passengers: int


# ---- NEW ENDPOINT: /search-business ----
@app.post("/search-business")
def search_business(params: SearchRequest):

    # Hardcoded sample result for MVP
    sample_result = [
        {
            "id": 1,
            "origin": params.origin,
            "destination": params.destination,
            "price": 1299,
            "airline": "Turkish Airlines",
            "outboundDate": params.earliestDeparture,
            "returnDate": params.latestDeparture,
            "durationHours": 14,
            "stops": 1
        },
        {
            "id": 2,
            "origin": params.origin,
            "destination": params.destination,
            "price": 1550,
            "airline": "Lufthansa",
            "outboundDate": params.earliestDeparture,
            "returnDate": params.latestDeparture,
            "durationHours": 10,
            "stops": 1
        }
    ]

    return {
        "status": "ok",
        "options": sample_result
    }
