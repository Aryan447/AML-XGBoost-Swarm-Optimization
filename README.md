![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-optimized-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-teal.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/build-passing-success.svg)

# AML Detection System

Anti-Money Laundering (AML) detection system using XGBoost with Swarm-Based Metaheuristic Optimization. This repository contains a production-ready FastAPI service for detecting suspicious financial transactions.

## Features

- 🚀 FastAPI-based REST API
- 🤖 XGBoost model optimized with Grey Wolf Optimization (GWO)
- 🐳 Docker support for easy deployment
- 📊 Real-time transaction risk scoring
- 🔒 Production-ready error handling and logging

## Project Structure

```
AML-XGBoost-Swarm-Optimization/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── core/            # Configuration
│   ├── schemas/         # Pydantic models
│   ├── services/        # Business logic (ModelService)
│   ├── static/          # Frontend assets
│   └── main.py          # FastAPI application
├── models/              # Trained model artifacts
├── scripts/             # Training scripts
├── tests/               # Test suite
└── requirements.txt     # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AML-XGBoost-Swarm-Optimization
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files exist**
   Make sure the `models/` directory contains:
   - `best_model_gwo.json`
   - `scaler.pkl`
   - `feature_columns.pkl`

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

6. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Frontend: http://localhost:8000/
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**
   ```bash
   docker build -t aml-api .
   docker run -p 8000:8000 -v ./models:/app/models aml-api
   ```

## API Usage

### Predict Transaction Risk

**Endpoint:** `POST /api/v1/predict`

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "is_laundering": 0,
  "risk_score": 0.2345,
  "risk_level": "LOW"
}
```

**Risk Levels:**
- `LOW`: risk_score ≤ 0.5
- `HIGH`: 0.5 < risk_score ≤ 0.8
- `CRITICAL`: risk_score > 0.8

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

## Environment Variables

- `MODEL_DIR`: Path to model directory (default: `/app/models`)

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Development

### Code Structure

- **API Layer** (`app/api/v1/endpoints.py`): Handles HTTP requests/responses
- **Service Layer** (`app/services/model_service.py`): Business logic and model inference
- **Schemas** (`app/schemas/transaction.py`): Request/response models
- **Configuration** (`app/core/config.py`): Application settings

### Key Features

- Singleton pattern for model service (model loaded once on startup)
- Comprehensive error handling with appropriate HTTP status codes
- Structured logging throughout the application
- Type hints for better code maintainability

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md) for details on our code of conduct and security policy.
