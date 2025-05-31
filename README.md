# DCA API Documentation

## Base URL
```
http://localhost:8000
# or with ngrok
https://your-ngrok-url.ngrok-free.app
```

---

## API Endpoints

### 1. GET /get_wells
**Purpose**: Get list of available wells

**Request**:
```
GET /get_wells
```

**Response**:
```json
{
  "wells": ["PKU00001-01", "PKU00002-01", "PKU00003-01", "PKU00005-01", "PKU00006-01"]
}
```

---

### 2. GET/POST /get_history
**Purpose**: Get historical production data

**URL Parameters**:
```
GET /get_history?well=PKU00001-01&start_date=2023-01-01&end_date=2023-12-31
```

**POST Body**:
```json
{
  "well": "PKU00001-01",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31"
}
```

**Response**:
```json
[
  {
    "Date": "2023-01-01",
    "Production": 150.5,
    "Fluid": 200.3,
    "JobCode": "PRF04"
  }
]
```

---

### 3. GET/POST /automatic_dca
**Purpose**: Perform decline curve analysis

**URL Parameters**:
```
GET /automatic_dca?well=PKU00001-01
```

**POST Body**:
```json
{
  "well": "PKU00001-01",
  "selected_data": [
    {
      "Date": "2023-01-01",
      "Production": 150.5,
      "Fluid": 200.3
    }
  ]
}
```

**Response**:
```json
{
  "Exponential": [150.5, 0.002],
  "Harmonic": [150.5, 0.001],
  "Hyperbolic": [150.5, 0.001, 1.2],
  "DeclineRate": {
    "Exponential": 73.0,
    "Harmonic": 36.5,
    "Hyperbolic": 36.5
  },
  "ActualData": [
    {
      "date": "2023-01-01",
      "value": 150.5,
      "fluid": 200.3
    }
  ],
  "StartDate": "2023-01-01",
  "EndDate": "2023-12-31"
}
```

---

### 4. GET/POST /predict_production
**Purpose**: Predict future production to economic limit

**URL Parameters**:
```
GET /predict_production?well=PKU00001-01&economic_limit=5
```

**POST Body**:
```json
{
  "well": "PKU00001-01",
  "economic_limit": 5,
  "selected_data": {
    "Date": "2023-06-15",
    "Production": 120.5
  }
}
```

**Response**:
```json
{
  "ExponentialPrediction": [
    {"date": "2023-06-16", "value": 119.8},
    {"date": "2023-06-17", "value": 119.1}
  ],
  "HarmonicPrediction": [
    {"date": "2023-06-16", "value": 119.9}
  ],
  "HyperbolicPrediction": [
    {"date": "2023-06-16", "value": 119.7}
  ]
}
```

---

### 5. GET/POST /predict_ml
**Purpose**: Machine learning based prediction

**URL Parameters**:
```
GET /predict_ml?elr=10
```

**POST Body**:
```json
{
  "elr": 10.0
}
```

**Response**:
```json
{
  "dates_actual": ["2023-01-01", "2023-01-02"],
  "actual": [150.5, 148.2],
  "predicted": [149.8, 147.9],
  "dates_extended": ["2023-12-31", "2024-01-01"],
  "extended_prediction": [12.5, 11.8],
  "elr_threshold": 10.0
}
```

---

## URL Parameter Examples

### Complete Analysis Workflow
```
# 1. Load well history
GET /get_history?well=PKU00001-01&start_date=2023-01-01&end_date=2023-12-31

# 2. Perform DCA analysis
GET /automatic_dca?well=PKU00001-01

# 3. Predict to economic limit
GET /predict_production?well=PKU00001-01&economic_limit=5

# 4. ML prediction
GET /predict_ml?elr=10
```

### Frontend URL States
```
# History view
/?well=PKU00001-01&start_date=2023-01-01&end_date=2023-12-31&view=history

# DCA analysis view
/?well=PKU00001-01&view=dca&elr=5

# Prediction view
/?well=PKU00001-01&view=prediction&elr=10

# ML view
/?well=PKU00001-01&view=ml&elr=8
```

## Error Responses

```json
{
  "error": "Well PKU00999-01 not found in dataset."
}
```

```json
{
  "error": "Data terlalu sedikit untuk analisis DCA."
}
```

```json
{
  "error": "Run 'Model Automate DCA' first to generate DCA Prediction."
}
```

---

## Notes

- All endpoints support both GET (URL params) and POST (JSON body)
- Date filters default to last 24 months if not specified
- DCA analysis requires minimum 2 data points
- Production prediction requires prior DCA analysis
- ML prediction works independently of DCA analysis
- ngrok tunneling requires `ngrok-skip-browser-warning: true` header
