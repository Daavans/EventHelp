"""Unit tests for the tools. These run without calling the API."""

from src.tools import check_ticket, join_waitlist, parking_availability, run_tool


def test_check_ticket_valid():
    result = check_ticket("RF100003")
    assert result["status"] == "valid"
    assert result["type"] == "day"
    assert "name" not in result  # holder's name must never be exposed


def test_check_ticket_normalizes_input():
    assert check_ticket(" rf100001 ")["status"] == "valid"


def test_check_ticket_invalid_format():
    assert "error" in check_ticket("12345")


def test_check_ticket_not_found():
    assert "error" in check_ticket("RF999999")


def test_parking_full_on_saturday():
    assert parking_availability("Saturday")["available_spots"] == 0


def test_parking_invalid_day():
    assert "error" in parking_availability("monday")


def test_join_waitlist():
    result = join_waitlist("Test User", "test@example.com", "weekend")
    assert result["success"] is True


def test_join_waitlist_rejects_bad_email():
    assert "error" in join_waitlist("Test User", "not-an-email", "weekend")


def test_join_waitlist_rejects_day_tickets():
    assert "error" in join_waitlist("Test User", "test@example.com", "day")


def test_run_tool_unknown_tool():
    assert "error" in run_tool("delete_all_tickets", {})


def test_run_tool_missing_arguments():
    assert "error" in run_tool("check_ticket", {})
