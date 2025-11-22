from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date

app = FastAPI()

# Allow communication between frontend and backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Later replace with your Base44 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Model for /search-business ---
class SearchBusinessRequest(BaseModel):
    origin: str
    destination: str
    earliestDeparture: date
    latestDeparture: date
    minStayDays: int
    maxStayDays: int
    maxPrice: float
    cabin: str
    passengers: int


# --- Existing Routes ---
@app.get("/")
def home():
    return {"message": "Flyvo backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# --- NEW MVP Endpoint ---
@app.post("/search-business")
async def search_business(payload: SearchBusinessRequest):
    """
    MVP version of the business class flight search.
    Later this will connect to the Amadeus API.
    For now it returns no_results so the frontend can respond correctly.
    """

    # This is just a stub for now
    return {
        "status": "no_results",
        "options": []
    }
