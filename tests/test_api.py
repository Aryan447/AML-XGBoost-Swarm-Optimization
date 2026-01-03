"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "error"]


def test_root_endpoint():
    """Test root endpoint serves frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"


def test_predict_endpoint_valid_request():
    """Test prediction endpoint with valid transaction data."""
    transaction_data = {
        "Timestamp": "2022/09/01 08:30",
        "From Bank": 123,
        "Account": "ACC001",
        "To Bank": 456,
        "Account.1": "ACC002",
        "Amount Received": 10000.0,
        "Receiving Currency": "USD",
        "Amount Paid": 10000.0,
        "Payment Currency": "USD",
        "Payment Format": "Wire"
    }
    
    response = client.post("/api/v1/predict", json=transaction_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "is_laundering" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert data["is_laundering"] in [0, 1]
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_level"] in ["LOW", "HIGH", "CRITICAL"]


def test_predict_endpoint_missing_fields():
    """Test prediction endpoint with missing required fields."""
    incomplete_data = {
        "Timestamp": "2022/09/01 08:30",
        "From Bank": 123
    }
    
    response = client.post("/api/v1/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_types():
    """Test prediction endpoint with invalid data types."""
    invalid_data = {
        "Timestamp": "2022/09/01 08:30",
        "From Bank": "not_a_number",  # Should be int
        "Account": "ACC001",
        "To Bank": 456,
        "Account.1": "ACC002",
        "Amount Received": 10000.0,
        "Receiving Currency": "USD",
        "Amount Paid": 10000.0,
        "Payment Currency": "USD",
        "Payment Format": "Wire"
    }
    
    response = client.post("/api/v1/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_negative_amounts():
    """Test prediction endpoint rejects negative amounts."""
    invalid_data = {
        "Timestamp": "2022/09/01 08:30",
        "From Bank": 123,
        "Account": "ACC001",
        "To Bank": 456,
        "Account.1": "ACC002",
        "Amount Received": -1000.0,  # Negative amount
        "Receiving Currency": "USD",
        "Amount Paid": 10000.0,
        "Payment Currency": "USD",
        "Payment Format": "Wire"
    }
    
    response = client.post("/api/v1/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_different_payment_formats():
    """Test prediction endpoint with different payment formats."""
    payment_formats = ["Cash", "Cheque", "ACH", "Credit Card", "Wire", "Bitcoin", "Reinvestment"]
    
    for payment_format in payment_formats:
        transaction_data = {
            "Timestamp": "2022/09/01 08:30",
            "From Bank": 123,
            "Account": "ACC001",
            "To Bank": 456,
            "Account.1": "ACC002",
            "Amount Received": 10000.0,
            "Receiving Currency": "USD",
            "Amount Paid": 10000.0,
            "Payment Currency": "USD",
            "Payment Format": payment_format
        }
        
        response = client.post("/api/v1/predict", json=transaction_data)
        assert response.status_code == 200, f"Failed for payment format: {payment_format}"
        data = response.json()
        assert "risk_score" in data
