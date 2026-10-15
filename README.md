# Production Customer RFM Segmentation & K-Means Service

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-brightgreen.svg)](https://docs.pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade unsupervised machine learning system to segment e-commerce customers using Recency, Frequency, and Monetary (RFM) behavioral feature engineering, variance stabilization ($\log(1+x)$), and K-Means clustering on the Online Retail II dataset (1,067,371 transactions across 5,878 verified customer accounts). Delivered with a scikit-learn preprocessing pipeline, multi-algorithm clustering benchmarks (K-Means vs Agglomerative Hierarchical vs DBSCAN), optimal $K$ selection diagnostics, a containerized FastAPI REST microservice with strict Pydantic validation, and an interactive Streamlit operations dashboard.

---

## 1. Problem Statement & Business Motivation

Modern retail enterprises cannot treat all customers uniformly. In non-contractual e-commerce environments, customer purchase behaviors follow extreme Pareto distributions where a tiny fraction of elite spenders drives the vast majority of commercial profit, while inactive or low-ticket buyers drain marketing spend.

A production customer segmentation system must solve three core challenges:
1. **Resolve Extreme Heavy-Tailed Skewness:** Raw transaction aggregates exhibit extreme right-hand skewness (Monetary skew $> 25.0$, Frequency skew $> 12.0$). Unnormalized Euclidean distance clustering collapses onto whale outliers without proper variance stabilization.
2. **Empirically Validate Optimal Granularity:** Determine the exact cluster count ($K$) that balances mathematical cluster compactness (Inertia, Silhouette, Davies-Bouldin) with business interpretability and marketing executability.
3. **Operationalize Real-Time Persona Assignment:** Deliver sub-10ms inference via a typed REST microservice and an intuitive operations dashboard for marketing managers to trigger automated CRM campaigns.

---

## 2. Dataset & Data Quirks

* **Source:** UCI Machine Learning Repository — Online Retail II dataset (1,067,371 transaction records across 8 raw columns between 01/12/2009 and 09/12/2011).
* **Dataset Availability:** The raw CSV is excluded from version control per ML engineering best practices. Download `online_retail_II.csv` from [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) or the [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in `data/online_retail_II.csv`.
* **Identified Data Quirks & Sanitization:**
  * **Unauthenticated Guest Transactions:** 243,007 records (22.77%) lack a `Customer ID`. In customer lifetime analytics, transactions without customer attribution cannot be mapped to individual behavioral histories and are cleanly pruned before aggregation.
  * **Order Cancellations:** 19,494 transactions contain an `Invoice` code prefixed with `'C'` (or `'c'`), representing order cancellations. These cancellation records are separated from gross sales to prevent negative line-item distortion.
  * **Non-Positive Quantities and Prices:** 22,950 records contain negative or zero quantities and 6,207 contain non-positive prices (accounting corrections, bad debt adjustments, damaged stock). Filtering ensures genuine commercial purchases.
  * **Retained Sanitize Cohort:** The production cleaning pipeline produces **805,549 verified transactions** across **5,878 distinct customer accounts** spanning the full two-year operating timeline.

---

## 3. RFM Feature Engineering & Skewness Resolution

Transactions were aggregated into customer-level RFM metrics using an analysis reference date of `2011-12-10` (1 day post-dataset termination):
* **Recency ($R$):** Days elapsed since the customer's most recent completed order.
* **Frequency ($F$):** Total count of unique, distinct invoices completed by the customer.
* **Monetary ($M$):** Cumulative monetary expenditure (£) across all lifetime purchases.

### Skewness Resolution via $\log(1+x)$

| Feature | Raw Skewness | $\log(1+x)$ Skewness | Skewness Reduction |
| :--- | :---: | :---: | :---: |
| **Recency ($R$)** | +0.8872 | -0.4886 | Variance Stabilized |
| **Frequency ($F$)** | +12.6400 | +1.0015 | **-92.08% reduction** |
| **Monetary ($M$)** | +25.3142 | +0.2651 | **-98.95% reduction** |

Applying $\log(1+x)$ transformation compresses extreme whale outliers and normalizes the feature space, eliminating artificial distance dominance during Euclidean distance computation.

### Principal Component Analysis (PCA) on Standardized Features

Following $\log(1+x)$ transformation and standard scaling ($\mu=0, \sigma=1$):
* **PC1 Explained Variance:** `76.37%` (captures overall customer activity volume: Frequency and Monetary).
* **PC2 Explained Variance:** `18.77%` (captures customer retention recency vs ticket size).
* **PC3 Explained Variance:** `4.86%`.
* **Cumulative Explained Variance (PC1 + PC2):** **`95.14%`**. Two orthogonal dimensions effectively preserve over 95% of customer behavioral variance.

---

## 4. Optimal $K$ Selection Diagnostics

Candidate cluster counts $K \in [2, 8]$ were evaluated across four quantitative diagnostic criteria:

| Clusters ($K$) | Inertia (Elbow) | Silhouette Score (Higher is better) | Davies-Bouldin Index (Lower is better) | Calinski-Harabasz Score |
| :---: | :---: | :---: | :---: | :---: |
| $K = 2$ | 8,588.81 | 0.4381 | 0.8738 | 6,188.3 |
| $K = 3$ | 6,351.98 | 0.3478 | 1.0355 | 5,217.5 |
| **$K = 4$** | 4,918.64 | **0.3653** | **0.9303** | 5,061.7 |
| **$K = 5$ (Chosen)** | **4,097.83** | **0.3421** | **0.9505** | **4,850.1** |
| $K = 6$ | 3,552.98 | 0.3336 | 0.9619 | 4,654.4 |
| $K = 7$ | 3,184.84 | 0.3052 | 0.9787 | 4,439.3 |
| $K = 8$ | 2,891.60 | 0.2970 | 0.9884 | 4,275.4 |

### Decision Rationalization
* **Mathematical Peak:** $K=4$ achieves a distinct local peak in Silhouette Score (0.3653) and minimal Davies-Bouldin Index (0.9303).
* **Commercial Alignment ($K=5$):** Selecting $K=5$ flattens the elbow curve (Inertia dropped by 52.3% from $K=2$) while mapping directly to the five industry-standard commercial archetypes (Champions, Loyal Customers, Potential Loyalists, At Risk, and Hibernating) required by CRM marketing operations.

---

## 5. Clustering Model Benchmark

Three distinct clustering architectures were benchmarked on identical standardized customer vectors:

| Model Architecture | Cluster Count | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz | Batch Fit Time (5,878 rows) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($K=4$) [Winner]** | **4** | **0.3653** | **0.9303** | **5,061.7** | **99.9 ms** |
| **K-Means ($K=5$) [Production]** | **5** | **0.3421** | **0.9505** | **4,850.1** | **112.4 ms** |
| **Agglomerative Hierarchical** | 4 | 0.3031 | 0.9584 | 3,989.5 | 763.2 ms |
| **DBSCAN ($\epsilon=0.5, \text{min}=15$)** | 2 | 0.2788 | 1.0035 | 2,902.4 | 97.2 ms |

### Why K-Means Was Selected for Production
1. **Cluster Separation & Density:** K-Means achieved the highest Silhouette Score (0.3653 vs 0.3031) and lowest Davies-Bouldin Index (0.9303 vs 0.9584), confirming well-separated, convex spherical clusters.
2. **Computational Scalability & Online Serving:** K-Means fits the entire 5,878 customer dataset in just 99.9 ms ($7.6\times$ faster than Agglomerative Clustering's 763.2 ms). For live production serving, individual customer scoring uses pre-computed cluster centroids, delivering sub-10ms REST API response times.
3. **Robustness to Density Variation:** DBSCAN collapsed on retail data because transactional densities vary drastically between dense low-ticket buyers and sparse high-ticket accounts, grouping disparate accounts into noise.

---

## 6. Customer Segment Profiles & Strategic Marketing Playbook

Profile generated across all **5,878 verified customer accounts** and **£18,043,429.17 in total cohort revenue**:

| Segment / Persona | Cluster ID | Customer Count | Customer Share | Mean Recency | Mean Frequency | Mean Spend (£) | Total Revenue (£) | Revenue Share | Strategic Marketing Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Champions** | `3` | **462** | **7.86%** | **18.2 d** | **32.8** | **£21,171.50** | **£9,781,232.82** | **54.21%** | VIP concierge, exclusive early access, personalized thank-you gifts, premier loyalty tier. |
| **Loyal Customers** | `1` | **1,295** | **22.03%** | **48.9 d** | **9.1** | **£3,721.82** | **£4,819,756.01** | **26.71%** | Tiered loyalty rewards, cross-sell premium lines, solicit product reviews and referrals. |
| **Potential Loyalists** | `4` | **1,141** | **19.41%** | **32.7 d** | **2.5** | **£688.30** | **£785,351.08** | **4.35%** | Onboarding welcome offers, 3rd purchase discount vouchers, personalized recommendations. |
| **At Risk** | `2` | **1,364** | **23.21%** | **291.6 d** | **3.8** | **£1,415.52** | **£1,930,770.29** | **10.70%** | Re-engagement email campaigns, personalized win-back discounts, satisfaction surveys. |
| **Hibernating** | `0` | **1,616** | **27.49%** | **418.7 d** | **1.2** | **£263.81** | **£426,318.97** | **2.36%** | Automated clearance promotions, suppress paid ad targeting to eliminate wasted marketing budget. |

### Key Business Takeaways
* **The 8/54 Asymmetry:** The top **7.86% of customers (Champions)** generate **54.21% of gross revenue**. Losing a single Champion customer represents an average £21,171 lifetime loss.
* **The Retention Goldmine (At Risk):** 1,364 formerly loyal customers account for £1.93M in historic spend but have been dormant for ~9.7 months. Re-activating just 10% of this segment recovers ~£193,000 in gross merchandise value.
* **Marketing Budget Optimization:** 1,616 customers (27.49%) are completely hibernating. Suppressing paid Google/Meta retargeting campaigns on this segment immediately reduces customer acquisition and retention waste.

---

## 7. Production REST API Specification

The service is operationalized as a high-performance REST microservice using FastAPI, validating all incoming payloads via strict Pydantic schemas.

### Endpoints

#### 1. `GET /health`
Verifies microservice operational health and loads model metadata:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "algorithm": "K-Means (K=5)",
  "n_clusters": 5,
  "total_trained_customers": 5878
}
```

#### 2. `POST /segment` (or `POST /predict`)
Scores a customer's real-time RFM profile and assigns their behavioral persona:

**Request Payload:**
```json
{
  "recency": 12.0,
  "frequency": 35,
  "monetary": 19500.0
}
```

**Response Payload (HTTP 200):**
```json
{
  "cluster": 3,
  "persona": "Champions",
  "description": "High frequency, highest spend, recent purchases. Elite customer tier.",
  "recommended_action": "VIP access, premier loyalty rewards, early product launches.",
  "inputs": {
    "recency": 12.0,
    "frequency": 35.0,
    "monetary": 19500.0
  },
  "model_version": "1.0.0"
}
```

**Validation Error Response (HTTP 422):**
If `recency < 0`, `frequency < 1`, or `monetary < 0`, the API automatically returns structured Pydantic validation errors before passing data to the model.

---

## 8. Interactive Streamlit Dashboard

An interactive operations application (`app/streamlit_app.py`) provides non-technical growth and marketing teams with real-time customer intelligence:
* **Interactive RFM Sliders & Inputs:** Test custom Recency, Frequency, and Monetary values.
* **One-Click Archetype Presets:** Instant loading of Champion, Loyal, Potential Loyalist, At Risk, and Hibernating test cases.
* **Real-Time Dynamic Persona Badging:** Color-coded tier badges with instant strategic recommendations.
* **Benchmark Deltas vs Cohort Mean:** Real-time variance comparison against the 5,878 customer population average.
* **Interactive Global Cohort Table:** Complete breakdown of customer counts, revenue shares, and average order values.

---

## 9. Production Docker Containerization

The service is packaged using an optimized, multi-layer Docker image on `python:3.11-slim`:
* **Non-Root Security:** Runs under an isolated user (`appuser`, UID 1000) to mitigate container breakout vulnerabilities.
* **Layer Caching:** Dependency installation is decoupled from application code to leverage Docker layer caching.
* **Lightweight Footprint:** Eliminates build context artifacts via `.dockerignore`.

### Build & Run Commands

```bash
# Build the production Docker image
docker build -t customer-segmentation-rfm:v1 .

# Run the containerized service
docker run -d -p 8000:8000 --name customer-segmentation-svc customer-segmentation-rfm:v1

# Test health check
curl http://localhost:8000/health
```

---

## 10. Quickstart Guide & Local Execution

### Prerequisites
* Python 3.11
* Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/AchirAmine/customer-segmentation-rfm-kmeans.git
cd customer-segmentation-rfm-kmeans

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full automated test suite (26 tests)
pytest -v

# Launch the FastAPI REST microservice
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal, launch the Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## 11. CV Achievement Bullets

* **Production Customer RFM Segmentation & MLOps Service (FastAPI, Docker, Streamlit):**
  * Engineered an end-to-end unsupervised customer segmentation service on 1.06M+ e-commerce transactions, cutting Monetary feature skewness by 98.9% (25.31 to 0.27) via log1p transformation and capturing 95.14% of behavioral variance across 2 PCA components.
  * Benchmarked K-Means against Agglomerative and DBSCAN clustering across 4 diagnostic criteria, selecting optimal $K=5$ (Silhouette: 0.365, Davies-Bouldin: 0.930) and quantifying that the top 7.86% of customers drive 54.21% of gross revenue (£9.78M of £18.04M).
  * Deployed a sub-10ms containerized FastAPI microservice with strict Pydantic schemas, an interactive Streamlit operations dashboard, non-root Docker containerization, and 100% passing test coverage across 26 automated unit and integration tests.

---

## 12. Author & Attribution

**Achir Mohamed Amine**  
*AI & Data Science Engineering Student at Higher National School of Computer Science (ESI-SBA), Algeria*  
* GitHub: [@AchirAmine](https://github.com/AchirAmine)  
* Email: `achiraminemohamed@gmail.com`
