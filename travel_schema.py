"""Unified schema for travel-document feature extraction.

One uploaded file is one document. That document may contain several passengers
and several bookings. Each flight, train, bus, hotel, car, or cruise is its own
item in a list, so a connecting itinerary is several segments rather than one
flattened trip.

Values are copied from the document. Dates keep the printed text plus an ISO
8601 value when the text can be read without guessing. A missing timezone stays
null. Money keeps the printed text plus the amount and currency when those are
explicit. Fields that are not on the document are null, and missing collections
are empty lists.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DocumentType(str, Enum):
    flight = "flight"
    boarding_pass = "boarding_pass"
    hotel = "hotel"
    train = "train"
    bus = "bus"
    car_rental = "car_rental"
    cruise = "cruise"
    itinerary = "itinerary"
    other = "other"
    unknown = "unknown"


class DateTimeValue(BaseModel):
    """Printed date or time, plus a normalized value when one is reliable."""

    model_config = ConfigDict(extra="ignore")

    original: str | None = None
    iso: str | None = None
    timezone: str | None = None


class MoneyValue(BaseModel):
    """Printed price, without converting the amount to a float."""

    model_config = ConfigDict(extra="ignore")

    original: str | None = None
    amount: str | None = None
    currency: str | None = None


class ContactDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)


class Passenger(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    passenger_type: str | None = None
    ticket_number: str | None = None
    seat: str | None = None
    coach: str | None = None


class FlightSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    airline: str | None = None
    flight_number: str | None = None
    departure_airport: str | None = None
    departure_code: str | None = None
    arrival_airport: str | None = None
    arrival_code: str | None = None
    departure: DateTimeValue | None = None
    arrival: DateTimeValue | None = None
    departure_terminal: str | None = None
    arrival_terminal: str | None = None
    departure_gate: str | None = None
    arrival_gate: str | None = None
    seat: str | None = None
    cabin_class: str | None = None
    baggage: str | None = None
    boarding_time: DateTimeValue | None = None
    status: str | None = None
    fare: MoneyValue | None = None


class HotelStay(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    address: str | None = None
    contact: ContactDetails = Field(default_factory=ContactDetails)
    check_in: DateTimeValue | None = None
    check_out: DateTimeValue | None = None
    nights: int | None = None
    rooms: str | None = None
    room_type: str | None = None
    guests: str | None = None
    guest_names: list[str] = Field(default_factory=list)
    booking_reference: str | None = None
    cancellation_policy: str | None = None
    payment_status: str | None = None
    total_price: MoneyValue | None = None
    amenities: list[str] = Field(default_factory=list)


class TrainSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operator: str | None = None
    train_name: str | None = None
    train_number: str | None = None
    departure_station: str | None = None
    arrival_station: str | None = None
    departure: DateTimeValue | None = None
    arrival: DateTimeValue | None = None
    coach: str | None = None
    seat: str | None = None
    travel_class: str | None = None
    platform: str | None = None


class BusSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operator: str | None = None
    service_number: str | None = None
    departure_location: str | None = None
    arrival_location: str | None = None
    departure: DateTimeValue | None = None
    arrival: DateTimeValue | None = None
    boarding_point: str | None = None
    dropping_point: str | None = None
    seat: str | None = None
    travel_class: str | None = None


class CarRental(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: str | None = None
    vehicle: str | None = None
    pickup_location: str | None = None
    dropoff_location: str | None = None
    pickup: DateTimeValue | None = None
    dropoff: DateTimeValue | None = None
    driver_name: str | None = None
    confirmation_number: str | None = None
    duration: str | None = None
    total_price: MoneyValue | None = None
    payment_status: str | None = None


class CruiseBooking(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cruise_line: str | None = None
    ship_name: str | None = None
    departure_port: str | None = None
    arrival_port: str | None = None
    departure: DateTimeValue | None = None
    arrival: DateTimeValue | None = None
    cabin: str | None = None
    confirmation_number: str | None = None


class TravelDocument(BaseModel):
    """Predictable top-level document produced by feature extraction."""

    model_config = ConfigDict(extra="ignore")

    document_type: DocumentType = DocumentType.unknown
    document_subtype: str | None = None
    document_title: str | None = None
    booking_references: list[str] = Field(default_factory=list)
    confirmation_numbers: list[str] = Field(default_factory=list)
    ticket_numbers: list[str] = Field(default_factory=list)
    booking_status: str | None = None
    booking_date: DateTimeValue | None = None
    issue_date: DateTimeValue | None = None
    provider_name: str | None = None
    provider_contact: ContactDetails = Field(default_factory=ContactDetails)
    passengers: list[Passenger] = Field(default_factory=list)
    currency: str | None = None
    total_price: MoneyValue | None = None
    payment_status: str | None = None
    flights: list[FlightSegment] = Field(default_factory=list)
    hotels: list[HotelStay] = Field(default_factory=list)
    trains: list[TrainSegment] = Field(default_factory=list)
    buses: list[BusSegment] = Field(default_factory=list)
    car_rentals: list[CarRental] = Field(default_factory=list)
    cruises: list[CruiseBooking] = Field(default_factory=list)
    baggage: str | None = None
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    extraction_status: Literal["complete", "partial", "unknown"] = "unknown"
    source_filename: str | None = None


_DATE_TIME_FORMATS = (
    "%d %b %Y %H:%M",
    "%d %b %Y %H:%M:%S",
    "%d %b %y %H:%M",
    "%d-%b-%Y %H:%M",
    "%d-%b-%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M",
)
_DATE_FORMATS = (
    "%d %b %Y",
    "%d %b %y",
    "%d-%b-%Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
)
_CURRENCY_CODES = ("INR", "USD", "EUR", "GBP", "AED", "SGD", "AUD", "CAD", "JPY", "CNY", "THB", "CHF")
_TIMEZONE_NAMES = ("UTC", "GMT", "IST", "PST", "EST", "CST", "MST")
_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:\d{2}|[+-]\d{4})")
_NAMED_ZONE_RE = re.compile(r"\b(" + "|".join(_TIMEZONE_NAMES) + r")\b")
_CURRENCY_RE = re.compile(r"\b(" + "|".join(_CURRENCY_CODES) + r")\b", re.I)
_AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WEEKDAY_RE = re.compile(r"^([A-Za-z]{3,9}),?\s+")
_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
_MONTH_WORDS = {datetime(2000, month, 1).strftime("%B").lower() for month in range(1, 13)}
_MONTH_WORDS.update(datetime(2000, month, 1).strftime("%b").lower() for month in range(1, 13))
_LEGACY_SUBTYPES = frozenset({"invoice", "visa", "package"})


def parse_datetime(value: str | None) -> DateTimeValue | None:
    """Keep the printed value. Set iso only when the calendar date is explicit."""
    if value is None:
        return None
    original = " ".join(str(value).split())
    if not original:
        return None

    timezone = None
    offset = _OFFSET_RE.search(original)
    named = _NAMED_ZONE_RE.search(original)
    if offset:
        timezone = offset.group(1)
    elif named:
        timezone = named.group(1)

    candidate = original
    if timezone:
        candidate = candidate.replace(timezone, " ")
    candidate = " ".join(candidate.replace("'", " ").split())
    weekday = _WEEKDAY_RE.match(candidate)
    if weekday and weekday.group(1).lower() not in _MONTH_WORDS:
        candidate = candidate[weekday.end() :]
    candidate = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", candidate, flags=re.I)

    if _TIME_ONLY_RE.fullmatch(candidate):
        return DateTimeValue(original=original, iso=None, timezone=timezone)

    for fmt in _DATE_TIME_FORMATS:
        parsed = _try_strptime(candidate, fmt)
        if parsed is None:
            continue
        iso = parsed.strftime("%Y-%m-%dT%H:%M")
        if timezone and re.fullmatch(r"[+-]\d{2}:\d{2}", timezone):
            iso = f"{iso}{timezone}"
        elif timezone == "Z":
            iso = f"{iso}Z"
        return DateTimeValue(original=original, iso=iso, timezone=timezone)

    for fmt in _DATE_FORMATS:
        parsed = _try_strptime(candidate, fmt)
        if parsed is None:
            continue
        return DateTimeValue(original=original, iso=parsed.strftime("%Y-%m-%d"), timezone=timezone)

    return DateTimeValue(original=original, iso=None, timezone=timezone)


def parse_money(value: str | None, currency_hint: str | None = None) -> MoneyValue | None:
    if value is None:
        return None
    original = " ".join(str(value).split())
    if not original:
        return None
    code_match = _CURRENCY_RE.search(original)
    currency = code_match.group(1).upper() if code_match else _currency_code(currency_hint)
    amount_match = _AMOUNT_RE.search(original)
    return MoneyValue(
        original=original,
        amount=amount_match.group(0) if amount_match else None,
        currency=currency,
    )


def _currency_code(value: str | None) -> str | None:
    if not value:
        return None
    match = _CURRENCY_RE.search(value)
    if match:
        return match.group(1).upper()
    upper = value.strip().upper()
    if upper in _CURRENCY_CODES:
        return upper
    return None


def _try_strptime(value: str, fmt: str) -> datetime | None:
    try:
        return datetime.strptime(value, fmt)
    except ValueError:
        return None


def _clean(value) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _string_list(values) -> list[str]:
    found: list[str] = []
    for value in values or []:
        text = _clean(value)
        if not text or text in found:
            continue
        found.append(text)
    return found


def _place_name(place) -> str | None:
    if not isinstance(place, dict):
        return _clean(place)
    return _clean(place.get("name"))


def _place_part(place, key: str) -> str | None:
    if not isinstance(place, dict):
        return None
    return _clean(place.get(key))


def _when(value, warnings: list[str], label: str) -> DateTimeValue | None:
    parsed = parse_datetime(_clean(value))
    if parsed is not None and parsed.iso is None:
        warnings.append(f"{label} could not be normalized: {parsed.original}")
    return parsed


def _nights(value) -> int | None:
    text = _clean(value)
    if text and text.isdigit():
        return int(text)
    return None


def _provider_name(document: TravelDocument) -> str | None:
    names: list[str] = []
    for flight in document.flights:
        if flight.airline:
            names.append(flight.airline)
    for hotel in document.hotels:
        if hotel.name:
            names.append(hotel.name)
    for train in document.trains:
        if train.operator:
            names.append(train.operator)
    for bus in document.buses:
        if bus.operator:
            names.append(bus.operator)
    for rental in document.car_rentals:
        if rental.company:
            names.append(rental.company)
    for cruise in document.cruises:
        if cruise.cruise_line:
            names.append(cruise.cruise_line)
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return unique[0]
    return None


def _status(document: TravelDocument) -> Literal["complete", "partial", "unknown"]:
    has_content = any(
        (
            document.passengers,
            document.flights,
            document.hotels,
            document.trains,
            document.buses,
            document.car_rentals,
            document.cruises,
            document.booking_references,
            document.confirmation_numbers,
            document.ticket_numbers,
        )
    )
    if not has_content:
        return "unknown"
    if document.warnings:
        return "partial"
    return "complete"


def _legacy_type(features: dict) -> tuple[DocumentType, str | None]:
    raw = _clean(features.get("document_type")) or "unknown"
    subtype = raw if raw in _LEGACY_SUBTYPES else None
    mapping = {
        "flight": DocumentType.flight,
        "boarding_pass": DocumentType.boarding_pass,
        "hotel": DocumentType.hotel,
        "train": DocumentType.train,
        "bus": DocumentType.bus,
        "car_rental": DocumentType.car_rental,
        "cruise": DocumentType.cruise,
        "itinerary": DocumentType.itinerary,
        "unknown": DocumentType.unknown,
    }
    return mapping.get(raw, DocumentType.other), subtype


def _first(value, fallback):
    return value if value else fallback


def to_travel_document(
    features: dict,
    source_text: str = "",
    source_filename: str | None = None,
) -> TravelDocument:
    """Map the model result onto the unified document schema and validate it."""
    if _is_unified(features):
        try:
            document = TravelDocument.model_validate(features)
        except ValidationError as exc:
            raise RuntimeError(f"Travel document failed validation: {exc}") from exc
        if source_filename and not document.source_filename:
            document.source_filename = source_filename
        return document

    warnings: list[str] = []
    price_block = features.get("price") if isinstance(features.get("price"), dict) else {}
    currency = _currency_code(price_block.get("currency"))

    passengers: list[Passenger] = []
    people = features.get("travelers") or features.get("passengers") or []
    for item in people:
        if not isinstance(item, dict):
            continue
        passengers.append(
            Passenger(
                name=_clean(item.get("name")),
                passenger_type=_clean(_first(item.get("role"), item.get("passenger_type"))),
                ticket_number=_clean(item.get("ticket_number")),
                seat=_clean(item.get("seat")),
                coach=_clean(item.get("coach")),
            )
        )
    passengers = [
        person
        for person in passengers
        if person.name or person.seat or person.ticket_number
    ]

    flights: list[FlightSegment] = []
    trains: list[TrainSegment] = []
    buses: list[BusSegment] = []
    cruises: list[CruiseBooking] = []
    for item in features.get("transport") or []:
        if not isinstance(item, dict):
            continue
        mode = (_clean(item.get("mode")) or "").lower()
        origin = item.get("from") if isinstance(item.get("from"), dict) else {}
        destination = item.get("to") if isinstance(item.get("to"), dict) else {}
        if mode == "flight":
            flights.append(
                FlightSegment(
                    airline=_clean(item.get("carrier")),
                    flight_number=_clean(item.get("service_number")),
                    departure_airport=_place_name(origin),
                    departure_code=_place_part(origin, "code"),
                    arrival_airport=_place_name(destination),
                    arrival_code=_place_part(destination, "code"),
                    departure=_when(origin.get("datetime"), warnings, "Flight departure"),
                    arrival=_when(destination.get("datetime"), warnings, "Flight arrival"),
                    departure_terminal=_place_part(origin, "terminal"),
                    arrival_terminal=_place_part(destination, "terminal"),
                    departure_gate=_place_part(origin, "gate"),
                    arrival_gate=_place_part(destination, "gate"),
                    seat=_clean(item.get("seat")),
                    cabin_class=_clean(_first(item.get("cabin_or_class"), item.get("cabin"))),
                    boarding_time=_when(item.get("boarding_time"), warnings, "Boarding time"),
                    status=_clean(item.get("status")),
                    fare=parse_money(item.get("fare"), currency),
                )
            )
        elif mode == "train":
            trains.append(
                TrainSegment(
                    operator=_clean(item.get("carrier")),
                    train_number=_clean(item.get("service_number")),
                    departure_station=_place_name(origin),
                    arrival_station=_place_name(destination),
                    departure=_when(origin.get("datetime"), warnings, "Train departure"),
                    arrival=_when(destination.get("datetime"), warnings, "Train arrival"),
                    coach=_clean(item.get("coach")),
                    seat=_clean(item.get("seat")),
                    travel_class=_clean(_first(item.get("cabin_or_class"), item.get("cabin"))),
                    platform=_clean(item.get("platform")),
                )
            )
        elif mode == "bus":
            buses.append(
                BusSegment(
                    operator=_clean(item.get("carrier")),
                    service_number=_clean(item.get("service_number")),
                    departure_location=_place_name(origin),
                    arrival_location=_place_name(destination),
                    departure=_when(origin.get("datetime"), warnings, "Bus departure"),
                    arrival=_when(destination.get("datetime"), warnings, "Bus arrival"),
                    seat=_clean(item.get("seat")),
                    travel_class=_clean(_first(item.get("cabin_or_class"), item.get("cabin"))),
                )
            )
        elif mode in {"ferry", "cruise"}:
            cruises.append(
                CruiseBooking(
                    cruise_line=_clean(item.get("carrier")),
                    departure_port=_place_name(origin),
                    arrival_port=_place_name(destination),
                    departure=_when(origin.get("datetime"), warnings, "Cruise departure"),
                    arrival=_when(destination.get("datetime"), warnings, "Cruise arrival"),
                    cabin=_clean(_first(item.get("cabin_or_class"), item.get("seat"))),
                )
            )

    hotels: list[HotelStay] = []
    stays = features.get("lodging") or features.get("hotels") or []
    for item in stays:
        if not isinstance(item, dict):
            continue
        hotels.append(
            HotelStay(
                name=_clean(item.get("name")),
                address=_clean(item.get("address")),
                check_in=_when(item.get("check_in"), warnings, "Hotel check-in"),
                check_out=_when(item.get("check_out"), warnings, "Hotel check-out"),
                nights=_nights(item.get("nights")),
                room_type=_clean(item.get("room_type")),
                guests=_clean(item.get("guests")),
                booking_reference=_clean(_first(item.get("confirmation_code"), item.get("booking_reference"))),
                cancellation_policy=_clean(_first(item.get("cancellation_policy"), item.get("cancellation"))),
                total_price=parse_money(_first(item.get("total"), item.get("total_price")), currency),
            )
        )

    rentals: list[CarRental] = []
    for item in features.get("car_rentals") or []:
        if not isinstance(item, dict):
            continue
        rentals.append(
            CarRental(
                company=_clean(item.get("company")),
                vehicle=_clean(item.get("vehicle")),
                pickup_location=_clean(item.get("pickup_location")),
                dropoff_location=_clean(item.get("dropoff_location")),
                pickup=_when(_first(item.get("pickup_datetime"), item.get("pickup")), warnings, "Car pickup"),
                dropoff=_when(_first(item.get("dropoff_datetime"), item.get("dropoff")), warnings, "Car drop-off"),
                confirmation_number=_clean(_first(item.get("confirmation_code"), item.get("confirmation_number"))),
                total_price=parse_money(_first(item.get("total"), item.get("fare")), currency),
            )
        )

    contacts = features.get("contacts") if isinstance(features.get("contacts"), dict) else {}
    declared_type, subtype = _legacy_type(features)
    categories = [
        name
        for name, items in (
            ("flight", flights),
            ("hotel", hotels),
            ("train", trains),
            ("bus", buses),
            ("car_rental", rentals),
            ("cruise", cruises),
        )
        if items
    ]
    if len(categories) > 1:
        document_type = DocumentType.itinerary
    elif len(categories) == 1:
        document_type = DocumentType(categories[0])
        if document_type == DocumentType.flight and "boarding pass" in source_text.lower():
            document_type = DocumentType.boarding_pass
    else:
        document_type = declared_type

    if len(hotels) == 1 and document_type == DocumentType.hotel:
        hotels[0].guest_names = [person.name for person in passengers if person.name]

    baggage = _clean(features.get("baggage"))
    if len(flights) == 1 and baggage and not flights[0].baggage:
        flights[0].baggage = baggage

    price = features.get("price") if isinstance(features.get("price"), dict) else {}
    try:
        document = TravelDocument(
            document_type=document_type,
            document_subtype=subtype or _clean(features.get("document_subtype")),
            document_title=_clean(features.get("document_title")),
            booking_references=_string_list(features.get("booking_references")),
            confirmation_numbers=_string_list(
                _first(features.get("confirmation_codes"), features.get("confirmation_numbers"))
            ),
            ticket_numbers=_string_list(features.get("ticket_numbers")),
            booking_status=_clean(features.get("booking_status")),
            booking_date=_when(features.get("booking_date"), warnings, "Booking date"),
            issue_date=_when(features.get("issue_date"), warnings, "Issue date"),
            provider_contact=ContactDetails(
                emails=_string_list(contacts.get("emails")),
                phones=_string_list(contacts.get("phones")),
            ),
            passengers=passengers,
            currency=currency,
            total_price=parse_money(price.get("total"), currency),
            payment_status=_clean(_first(price.get("payment_status"), features.get("payment_status"))),
            flights=flights,
            hotels=hotels,
            trains=trains,
            buses=buses,
            car_rentals=rentals,
            cruises=cruises,
            baggage=baggage,
            notes=_string_list(_first(features.get("important_notes"), features.get("notes"))),
            warnings=warnings,
            source_filename=source_filename,
        )
    except ValidationError as exc:
        raise RuntimeError(f"Travel document failed validation: {exc}") from exc
    document.provider_name = _provider_name(document)
    document.extraction_status = _status(document)
    try:
        return TravelDocument.model_validate(document.model_dump())
    except ValidationError as exc:
        raise RuntimeError(f"Travel document failed validation: {exc}") from exc


def _is_unified(features: dict) -> bool:
    return "flights" in features or "hotels" in features or "trains" in features or "buses" in features
