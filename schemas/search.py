"""schemas/search.py - Pydantic models for search, jobs, and flight options."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CabinClass(str, Enum):
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"


class CabinSummary(str, Enum):
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class SearchParams(BaseModel):
    origin: str
    destination: str
    user_external_id: Optional[str] = None
    earliestDeparture: date
    latestDeparture: date
    minStayDays: int
    maxStayDays: int
    nights: Optional[int] = None
    maxPrice: Optional[float] = None
    cabin: str = "BUSINESS"
    passengers: int = 1
    stopsFilter: Optional[List[int]] = None

    maxOffersPerPair: int = 300
    maxOffersTotal: int = 10000
    maxDatePairs: int = 60


class SearchJob(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    params: SearchParams
    total_pairs: int = 0
    processed_pairs: int = 0
    error: Optional[str] = None


class FlightOption(BaseModel):
    id: str

    # Provider identity — used for checkout/booking handoff
    provider: Optional[str] = None
    providerSessionId: Optional[str] = None
    providerRecommendationId: Optional[str] = None

    airline: str
    airlineCode: Optional[str] = None
    price: float
    currency: str
    departureDate: str
    returnDate: str
    stops: int

    cabinSummary: Optional[CabinSummary] = None
    cabinHighest: Optional[CabinClass] = None
    cabinByDirection: Optional[Dict[str, Optional[CabinSummary]]] = None

    durationMinutes: int
    totalDurationMinutes: Optional[int] = None
    duration: Optional[str] = None

    origin: Optional[str] = None
    destination: Optional[str] = None
    originAirport: Optional[str] = None
    destinationAirport: Optional[str] = None

    stopoverCodes: Optional[List[str]] = None
    stopoverAirports: Optional[List[str]] = None

    outboundSegments: Optional[List[Dict[str, Any]]] = None
    returnSegments: Optional[List[Dict[str, Any]]] = None

    aircraftCodes: Optional[List[str]] = None
    aircraftNames: Optional[List[str]] = None

    bookingUrl: Optional[str] = None
    url: Optional[str] = None


class SearchStartResponse(BaseModel):
    status: JobStatus
    mode: str
    jobId: str


class SearchStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    processedPairs: int
    totalPairs: int
    progress: float
    error: Optional[str]
    previewCount: int
    previewOptions: List[FlightOption]
    elapsedSeconds: Optional[float] = None
    estimatedTotalSeconds: Optional[float] = None
    estimatedProgressPct: Optional[float] = None


class SearchResultsResponse(BaseModel):
    jobId: str
    status: JobStatus
    totalResults: int
    offset: int
    limit: int
    options: List[FlightOption]
