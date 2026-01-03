# 🦅 AML Detection System

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-Accelerated-blueviolet.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-teal.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Anti-Money Laundering (AML) Detection** powered by **XGBoost** and **Swarm Intelligence**, optimized for ultra-low latency inference using **ONNX Runtime**. This system detects suspicious financial transactions with high accuracy and minimal resource footprint.

## ⚡ Key Features

- **🚀 High Performance**: Sub-millisecond inference using ONNX Runtime.
- **☁️ production Ready**: Designed for serverless (Vercel) and containerized environments.
- **🛡️ Secure & Scalable**: FastAPI backend with Pydantic validation and robust error handling.
- **🧠 Advanced AI**: XGBoost classifier tuned via Grey Wolf Optimization (GWO).

---

## 🏗️ Architecture

The system uses a split-optimization strategy to deliver heavy ML capabilities in a lightweight package:

- **Training**: Models trained with XGBoost and optimized using Swarm Intelligence (GWO).
- **Inference**: Models converted to **ONNX** format for portable, dependency-free execution (`< 100MB` deployment).
- **API**: FastAPI provides a clean REST interface for real-time predictions.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)

### 🛠️ Local Installation

1. **Clone & Setup**
   ```bash
   git clone https://github.com/Aryan447/AML-XGBoost-Swarm-Optimization.git
   cd AML-XGBoost-Swarm-Optimization
   
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```
   Access documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 🐳 Docker Deployment

Run the complete stack with a single command:

```bash
docker-compose up --build
```

Or build manually:
```bash
docker build -t aml-api .
docker run -p 8000:8000 aml-api
```

---

## 🌐 API Reference

### 🔍 Predict Risk
`POST /api/v1/predict`

**Payload:**
```json
{
  "Timestamp": "2024-01-01 10:00:00",
  "From Bank": 10,
  "Account": "ACCX99",
  "To Bank": 12,
  "Account.1": "ACCY88",
  "Amount Received": 50000.0,
  "Receiving Currency": "USD",
  "Amount Paid": 50000.0,
  "Payment Currency": "USD",
  "Payment Format": "Wire",
  "Is Laundering": 0
}
```

**Response:**
```json
{
  "is_laundering": 0,
  "risk_score": 0.045,
  "risk_level": "LOW"
}
```

### 💓 Health Check
`GET /health`
Returns system status and model readiness.

---

## 📂 Project Structure

```
.
├── app/
│   ├── api/          # Endpoints
│   ├── services/     # Inference Logic (ONNX)
│   └── main.py       # App Entrypoint
├── models/           # Optimization Artifacts (.onnx)
├── public/           # Static Assets
└── tests/            # Pytest Suite
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
