# Learning Log — Retail Customer RFM Segmentation & K-Means Service

## S1 — Project Setup & Environment Isolation (2026-10-02)

**What I did:** Created the production src package layout, configured a dedicated Python 3.11 virtual environment with all scikit-learn, FastAPI, Streamlit, and pytest dependencies, isolated datasets and model binaries with a robust .gitignore, connected the GitHub remote repository, and initialized the project foundation.

**What I learned:** Isolating virtual environments and linking local packages via editable .pth configurations prevents version conflicts and avoids redundant package downloads. A clean src package layout strictly decouples core business logic from test harnesses and deployment entry points, while .gitignore safeguards repository performance and client transaction confidentiality.

## S2 — Data Cleaning Pipeline & Transaction Anomaly Resolution (2026-10-03)

**What I did:** Analyzed 1,067,371 raw transaction records from Online Retail II across 8 columns, identified and isolated 19,494 cancellation invoices (invoices prefixed with 'C'), pruned 22,950 non-positive quantities and 6,207 zero/negative unit prices, resolved 243,007 missing Customer IDs (22.77% unauthenticated transactions), and produced a sanitized cohort of 805,549 verified transactions across 5,878 distinct customers supported by 4 unit tests.

**What I learned:** Unsupervised customer segmentation requires individual identity attribution; transactions lacking Customer IDs cannot be assigned to customer lifetime journeys and must be cleanly filtered before aggregation. Filtering cancellations and negative line items ensures that monetary features reflect net commercial volume rather than return distortion.

## S3 — RFM Feature Engineering & Skewness Resolution (2026-10-05)

**What I did:** Engineered customer-level Recency (days relative to reference date 2011-12-10), Frequency (unique invoice count), and Monetary (total spend in GBP) features across 5,878 customers, quantified extreme right-hand distribution skewness (Frequency skew: 12.64, Monetary skew: 25.31), and resolved distributional asymmetry via log1p transformation, verified by 3 automated unit tests.

**What I learned:** E-commerce spend and transaction frequencies follow heavy-tailed Pareto distributions where a tiny percentage of whale buyers distorts linear distance metrics. Applying log1p transformation stabilized feature variance and reduced Frequency skewness from 12.64 to 1.00 (-92.1%) and Monetary skewness from 25.31 to 0.27 (-98.9%), creating an isotropic feature space essential for distance-based clustering.

## S4 — Preprocessing Pipeline & Principal Component Analysis (2026-10-06)

**What I did:** Engineered a scikit-learn compliant Log1pTransformer and comparative preprocessing pipelines benchmarking StandardScaler against RobustScaler, and conducted Principal Component Analysis (PCA) on the standardized customer feature vectors, validated by 4 unit tests.

**What I learned:** Standard scaling on log1p-transformed RFM features successfully centered all dimensions to zero mean and unit variance. PCA decomposition revealed that the first two principal components capture 95.14% of the total cumulative variance (PC1: 76.37%, PC2: 18.77%), confirming that customer behavior in this cohort is predominantly driven by engagement volume (F & M) alongside purchase recency.

## S5 — Optimal K Selection Diagnostics (2026-10-08)

**What I did:** Executed clustering diagnostics across candidate cluster counts $K \in [2, 8]$ using Inertia (Elbow Method), Silhouette Coefficient, Davies-Bouldin Index, and Calinski-Harabasz score on normalized customer feature vectors, supported by automated evaluation tests.

**What I learned:** While mathematical inertia monotonically decreases from 8,588.81 ($K=2$) to 2,891.60 ($K=8$), evaluation metrics reveal clear structure: $K=4$ achieves a distinct local peak in Silhouette Score (0.3653) and minimal Davies-Bouldin (0.9303), while $K=5$ flattens the elbow slope (Inertia: 4,097.83, Silhouette: 0.3421) and delivers superior commercial granularity for traditional 5-tier RFM persona mapping.

## S6 — Clustering Model Comparative Benchmark (2026-10-09)

**What I did:** Benchmarked K-Means ($K=4/5$) against Agglomerative Hierarchical Clustering and DBSCAN density clustering across Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz score, and batch execution time on the 5,878 customer dataset, verified by automated unit tests.

**What I learned:** K-Means decisively outperformed alternative architectures: achieving the highest Silhouette Score (0.3653 vs 0.3031 for Agglomerative and 0.2788 for DBSCAN), lowest Davies-Bouldin index (0.9303 vs 0.9584 and 1.0035), and 7.6x faster batch fit speed (99.9 ms vs 763.2 ms). For online serving, evaluating Euclidean distance to pre-computed centroids yields sub-10ms API responses. DBSCAN struggled with varying spatial densities across retail spenders, whereas K-Means formed coherent convex clusters.

## S7 — Cluster Profiling & Strategic Persona Mapping (2026-10-10)

**What I did:** Profiled the 5 customer clusters by computing mean/median Recency, Frequency, Monetary spend, customer cohort share, and cumulative revenue contribution, mapping each cluster to an actionable commercial archetype (Champions, Loyal Customers, Potential Loyalists, At Risk, Hibernating).

**What I learned:** Quantifying revenue contribution revealed striking commercial asymmetry: Champions account for only 7.86% of customers (462 accounts) but generate 54.21% of total cohort revenue (£9.78M of £18.04M), with an average spend of £21,171.50 and 32.8 orders. Conversely, Hibernating customers comprise 27.49% of accounts (1,616 accounts) but contribute just 2.36% of revenue (£426K), justifying immediate marketing budget reallocation towards retention and VIP loyalty programs.

## S8 — Atomic Model Serialization & Inference Engine (2026-10-12)

**What I did:** Serialized the complete fitted preprocessing pipeline, K-Means model, cluster-persona dictionary, persona definitions, and cohort baseline metrics into a single atomic joblib artifact (models/model_v1.joblib, 26.7 KB), built a production SegmentPredictor class, and authored 5 automated unit tests covering single and batch predictions and input error boundaries.

**What I learned:** Coupling feature transformation logic and cluster assignment into a single serialized bundle prevents training-serving skew and guarantees that real-time inference inputs undergo identical log1p scaling and standardization as the training set without code duplication.

## S9 — Production FastAPI Microservice & Schema Validation (2026-10-13)

**What I did:** Developed high-throughput GET /health, POST /segment, and POST /predict REST endpoints using FastAPI, enforced strict numerical boundaries (non-negative Recency/Monetary, positive Frequency) using Pydantic schemas, and authored 5 automated integration tests using TestClient.

**What I learned:** Pydantic type validation protects microservices at the network boundary, returning standard HTTP 422 Unprocessable Entity responses for malformed or out-of-bounds payloads before corrupt data can trigger downstream pipeline exceptions.

## S10 — Production Docker Containerization & Security Hardening (2026-10-14)

**What I did:** Authored a production-grade multi-layer Dockerfile on python:3.11-slim utilizing a dedicated non-root security user (appuser, UID 1000), configured .dockerignore to eliminate build context bloat, and verified container port exposure and runtime command execution.

**What I learned:** Structuring container layers so that requirements installation precedes application code copying maximizes Docker build cache utilization, while enforcing non-root execution prevents container breakout privileges in production cloud orchestrators.

## S11 — Interactive Streamlit Dashboard, Documentation & Career Deliverables (2026-10-15)

**What I did:** Built an interactive Streamlit operations dashboard featuring real-time customer segmentation, interactive RFM sliders, archetype presets, cohort baseline comparisons, and full cohort profiling tables, completed the comprehensive README.md, documented production architecture, and formulated 3 quantifiable CV achievement bullets.

**What I learned:** A mature machine learning system communicates both technical rigor and commercial impact: demonstrating that 7.86% of customers generate 54.21% of company revenue provides executive decision-makers with clear, actionable ROI that far exceeds raw academic metrics.
