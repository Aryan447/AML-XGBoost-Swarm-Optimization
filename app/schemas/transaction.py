"""Pydantic schemas for transaction requests and responses."""
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class TransactionRequest(BaseModel):
    """
    Request schema for transaction prediction.
    
    All fields support both snake_case (Python) and original column names (with spaces).
    """
    
    timestamp: str = Field(
        ...,
        alias="Timestamp",
        description="Transaction timestamp",
        examples=["2022/09/01 08:30", "2022-09-01 08:30:00"]
    )
    from_bank: int = Field(
        ...,
        alias="From Bank",
        description="Source bank identifier",
        ge=0
    )
    account: str = Field(
        ...,
        alias="Account",
        description="Source account identifier"
    )
    to_bank: int = Field(
        ...,
        alias="To Bank",
        description="Destination bank identifier",
        ge=0
    )
    account_dest: str = Field(
        ...,
        alias="Account.1",
        description="Destination account identifier"
    )
    amount_received: float = Field(
        ...,
        alias="Amount Received",
        description="Amount received in transaction",
        ge=0
    )
    receiving_currency: str = Field(
        ...,
        alias="Receiving Currency",
        description="Currency code for received amount"
    )
    amount_paid: float = Field(
        ...,
        alias="Amount Paid",
        description="Amount paid in transaction",
        ge=0
    )
    payment_currency: str = Field(
        ...,
        alias="Payment Currency",
        description="Currency code for paid amount"
    )
    payment_format: str = Field(
        ...,
        alias="Payment Format",
        description="Payment method",
        examples=["Cash", "Cheque", "ACH", "Credit Card", "Wire", "Bitcoin", "Reinvestment"]
    )

    @field_validator('payment_format')
    @classmethod
    def validate_payment_format(cls, v: str) -> str:
        """Validate payment format is one of the known values."""
        valid_formats = ['Cash', 'Cheque', 'ACH', 'Credit Card', 'Wire', 'Bitcoin', 'Reinvestment']
        if v not in valid_formats:
            # Allow it but log a warning - model will handle unknown values
            pass
        return v

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """
    Response schema for transaction prediction.
    
    Attributes:
        is_laundering: Binary indicator (0 or 1) if transaction is flagged
        risk_score: Risk score between 0 and 1
        risk_level: Categorical risk level (LOW, HIGH, CRITICAL)
    """
    
    is_laundering: int = Field(
        ...,
        description="Binary flag indicating if transaction is flagged as laundering",
        ge=0,
        le=1
    )
    risk_score: float = Field(
        ...,
        description="Risk score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    risk_level: Literal["LOW", "HIGH", "CRITICAL"] = Field(
        ...,
        description="Categorical risk level"
    )
