"""Tools the agent can call, backed by mock data, plus their definitions for the model."""

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "tickets.json"
DATA = json.loads(DATA_PATH.read_text(encoding="utf-8"))

TICKET_NUMBER_PATTERN = re.compile(r"^RF\d{6}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
FESTIVAL_DAYS = {"friday", "saturday", "sunday"}
SOLD_OUT_TICKET_TYPES = {"weekend", "camping"}

# In-memory waitlist: resets when the app restarts (fine for a demo).
waitlist: list[dict] = []


def check_ticket(ticket_number: str) -> dict:
    """Look up the type and status of a ticket."""
    ticket_number = ticket_number.strip().upper()
    # Guardrail: validate the format in code before looking anything up.
    if not TICKET_NUMBER_PATTERN.match(ticket_number):
        return {"error": "Invalid ticket number. Ticket numbers look like RF123456."}

    for ticket in DATA["tickets"]:
        if ticket["ticket_number"] == ticket_number:
            # Privacy: never return the ticket holder's name to the model.
            result = {"ticket_number": ticket_number, "type": ticket["type"], "status": ticket["status"]}
            if "day" in ticket:
                result["day"] = ticket["day"]
            return result
    return {"error": f"Ticket {ticket_number} was not found."}


def parking_availability(day: str) -> dict:
    """Return the number of free parking spots on a festival day."""
    day = day.strip().lower()
    if day not in FESTIVAL_DAYS:
        return {"error": "The festival days are friday, saturday and sunday."}
    parking = DATA["parking"][day]
    return {"day": day, "available_spots": parking["available"], "total_spots": parking["total"]}


def join_waitlist(name: str, email: str, ticket_type: str) -> dict:
    """Add someone to the waitlist for a sold-out ticket type."""
    ticket_type = ticket_type.strip().lower()
    if not name.strip():
        return {"error": "A name is required."}
    if not EMAIL_PATTERN.match(email.strip()):
        return {"error": "That email address does not look valid."}
    if ticket_type not in SOLD_OUT_TICKET_TYPES:
        return {"error": "There is only a waitlist for weekend and camping tickets. Day tickets are still on sale."}

    waitlist.append({"name": name.strip(), "email": email.strip(), "ticket_type": ticket_type})
    return {"success": True, "ticket_type": ticket_type, "position": len(waitlist)}


# Maps the name the model uses to the Python function we run.
TOOLS = {
    "check_ticket": check_ticket,
    "parking_availability": parking_availability,
    "join_waitlist": join_waitlist,
}

# Descriptions of the tools in JSON Schema, so the model knows when and how to call them.
TOOL_DEFINITIONS = [
    {
        "name": "check_ticket",
        "description": "Check the type and status (valid, used or cancelled) of a festival ticket.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string", "description": "Ticket number, e.g. RF123456."},
            },
            "required": ["ticket_number"],
        },
    },
    {
        "name": "parking_availability",
        "description": "Get the number of free parking spots at P-Waalstrand on a festival day.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "enum": ["friday", "saturday", "sunday"]},
            },
            "required": ["day"],
        },
    },
    {
        "name": "join_waitlist",
        "description": "Put a visitor on the waitlist for sold-out weekend or camping tickets. "
        "Only call this when the visitor has given their name and email address.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name of the visitor."},
                "email": {"type": "string", "description": "Email address of the visitor."},
                "ticket_type": {"type": "string", "enum": ["weekend", "camping"]},
            },
            "required": ["name", "email", "ticket_type"],
        },
    },
]


def run_tool(name: str, args: dict) -> dict:
    """Execute the tool the model asked for. Errors are returned to the model, not raised."""
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOLS[name](**args)
    except TypeError as e:  # wrong or missing arguments
        return {"error": f"Invalid arguments for {name}: {e}"}
