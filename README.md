# 🍔 Food Delivery Time Prediction — Data Analysis & Visualization

An end-to-end data analysis project on a food delivery logistics dataset, combining **Python (Pandas/Seaborn/Matplotlib)** for data cleaning and EDA, **KNIME** for machine learning and clustering workflows, and **Tableau** for interactive geographic and business dashboards.

---

## 1. Project Overview

The project analyzes a synthetic dataset of ~230 food delivery orders to understand what drives delivery time, tips, and service quality in an urban food-delivery setting (in the style of platforms like Zomato, Swiggy, UberEats, and DoorDash). The workflow spans:

- **Data cleaning & preprocessing** in Python (handling injected null values and invalid entries, column by column)
- **Exploratory Data Analysis (EDA)** — descriptive statistics, correlation analysis, outlier detection, and distribution analysis in Python
- **Machine learning experiments** (classification, regression, clustering) built visually as a **KNIME** workflow
- **Interactive dashboards** built in **Tableau**, focused on geographic delivery patterns and business filters

## 2. Problem Statement

Delivery time reliability is a key driver of customer satisfaction and operational cost in food delivery platforms. The project investigates which order-, restaurant-, and delivery-partner-level factors (weather, traffic, distance, ratings, experience, priority, etc.) are associated with delivery time, and builds exploratory groundwork for predicting it.

## 3. Objectives

- Develop a method to understand and predict food delivery time using historical order data.
- Analyze the influence of weather, traffic, location, ratings, cost, order priority, and delivery-person experience on delivery performance.
- Identify operational bottlenecks and surface recommendations for efficiency.
- Support stakeholders (customers, delivery partners, restaurants, platform managers, data science and customer service teams, regulators) with data-driven, visual findings.

## 4. Dataset

**Source file:** `Food_Delivery_Time_Prediction.csv` — 230 orders × 15 columns.

| Feature | Description |
|---|---|
| `Order_ID` | Unique identifier for each delivery |
| `Customer_Location` | Latitude/longitude of the customer |
| `Restaurant_Location` | Latitude/longitude of the restaurant |
| `Distance` | Distance between restaurant and customer |
| `Weather_Conditions` | Weather during delivery (Rainy, Cloudy, Snowy, Sunny) |
| `Traffic_Conditions` | Traffic level (Low, Medium, High) |
| `Delivery_Person_Experience` | Delivery person's experience |
| `Order_Priority` | Order urgency (Low, Medium, High) |
| `Order_Time` | Time of day the order was placed (Morning, Afternoon, Evening, Night) |
| `Vehicle_Type` | Delivery vehicle (Bike, Car, Bicycle) |
| `Restaurant_Rating` | Restaurant's rating |
| `Customer_Rating` | Customer's rating |
| `Delivery_Time` | Recorded delivery time — **target variable for regression** |
| `Order_Cost` | Total order cost |
| `Tip_Amount` | Tip paid by the customer |

**Dataset variants included in the repo:**

| File | Rows × Cols | Purpose |
|---|---|---|
| `Food_Delivery_Time_Prediction.csv` | 230 × 15 | Original clean base dataset |
| `generated_null_values.ipynb` | — | Notebook that programmatically injects a controlled number of null/invalid values per column into the base dataset (seeded with `np.random.seed(42)`), used to simulate real-world messy data |
| `Food_Delivery_Time_Prediction_custom_nulls.csv` | 230 × 15 | Output of the above — the "dirty" dataset actually used as the input for cleaning in both the notebook and the KNIME workflow |
| `Food_Delivery_Time_Prediction_without_null_values.csv` | 217 × 15 | Fully cleaned dataset produced by the notebook's preprocessing pipeline |
| `Food_Delivery_Time_Location_Added_Only.csv` | 217 × 19 | Cleaned dataset with `Customer_Location`/`Restaurant_Location` parsed into separate `Customer_Latitude`, `Customer_Longitude`, `Restaurant_Latitude`, `Restaurant_Longitude` columns — used as the Tableau data source |

## 5. Tools & Technologies

| Tool | Role in the project |
|---|---|
| **Python** (Pandas, NumPy, SciPy, Seaborn, Matplotlib) | Data cleaning, descriptive statistics, correlation analysis, outlier/IQR analysis, distribution plots, geographic scatter plot |
| **Jupyter Notebook** | `Innovative_Assignment.ipynb` (main analysis, 104 cells) and `generated_null_values.ipynb` (synthetic null injection) |
| **KNIME Analytics Platform** (v5.7) | Visual workflow for missing-value handling, EDA visualizations, K-Means clustering, and 5 supervised-learning experiments (Decision Tree, Naive Bayes, SVM, KNN, Linear Regression, Random Forest) |
| **Tableau** | Interactive geographic maps, heatmaps, and business-metric dashboards with parameters and calculated fields |
| **Microsoft Word / PDF** | `DAV Innovative Assignment.docx` / `.pdf` — the full written report documenting methodology, plots, and stakeholder-level findings |

## 6. Data Preprocessing

All cleaning is performed in **`Innovative_Assignment.ipynb`** on `Food_Delivery_Time_Prediction_custom_nulls.csv`, column by column:

- **`Order_ID`** — dropped rows with null `Order_ID`; dropped duplicate `Order_ID`s (keeping the first); dropped rows where `Order_ID` didn't match the pattern `ORD####`
- **`Customer_Location` / `Restaurant_Location`** — dropped rows with nulls (coordinates can't be reliably imputed)
- **`Distance`** — missing values recalculated from customer/restaurant coordinates using the **Haversine formula**; negative distances corrected by taking the absolute value
- **`Weather_Conditions`** — nulls filled, and any value outside `{Rainy, Cloudy, Snowy, Sunny}` replaced with `Cloudy`
- **`Traffic_Conditions`** — nulls filled with `Medium`; invalid values (outside `{Low, Medium, High}`) replaced with `Medium`
- **`Delivery_Person_Experience`** — nulls and non-positive values replaced with the mean of valid (positive) experience values
- **`Order_Priority`** — nulls and invalid values replaced with `Medium`
- **`Order_Time`** — nulls and invalid values replaced with `Night`
- **`Vehicle_Type`** — nulls and invalid values replaced with `Bike`
- **`Restaurant_Rating`** / **`Customer_Rating`** — nulls filled with the column mean; out-of-range values (outside 0–5) replaced with the mean of valid ratings
- **`Delivery_Time`**, **`Order_Cost`**, **`Tip_Amount`** — nulls filled with the column mean; negative values corrected via absolute value

The cleaned result is exported as **`Food_Delivery_Time_Prediction_without_null_values.csv`** (217 rows), which is then used downstream by both the KNIME workflow and the Tableau dashboards. Categorical label-encoding mappings (weather, traffic, priority, order time, vehicle type → integers) were drafted in the notebook but left commented out, so the exported CSV keeps the original categorical labels.

## 7. Exploratory Data Analysis

Computed directly in the notebook using `numpy`/`scipy.stats` on the cleaned data:

| Column | Mean | Median | Mode | Variance |
|---|---|---|---|---|
| Distance | 45.64 | 12.20 | 6.09 | 20,344.84 |
| Delivery_Person_Experience | 5.24 | 5.24 | 2.00 | 8.16 |
| Restaurant_Rating | 3.80 | 3.80 | 3.80 | 8.16 |
| Customer_Rating | 3.67 | 3.67 | 3.67 | 0.44 |
| Delivery_Time | 76.13 | 72.91 | 65.56 | 1,859.70 |
| Order_Cost | 1,180.87 | 1,054.82 | 1,026.39 | 690,518.51 |
| Tip_Amount | 50.33 | 46.48 | 46.48 | 1,652.39 |

**Correlation with `Delivery_Time`** (Pearson, on cleaned numeric columns):

```
Restaurant_Rating            -0.047
Distance                     -0.023
Order_Cost                   -0.015
Customer_Rating              -0.003
Delivery_Person_Experience    0.064
Tip_Amount                    0.107
```

All linear correlations with the target are weak, which the project report interprets as evidence that delivery time is driven by **non-linear, interacting effects** rather than any single strong linear predictor — motivating the multi-model KNIME experiments described in Section 9.

**Outlier / IQR analysis** (1.5×IQR rule) was computed for every numeric column, e.g.:

| Column | Q1 | Median | Q3 | IQR | Lower Bound | Upper Bound |
|---|---|---|---|---|---|---|
| Distance | 7.09 | 12.20 | 20.25 | 13.16 | -12.65 | 39.99 |
| Delivery_Time | 47.46 | 72.91 | 98.89 | 51.43 | -29.69 | 176.04 |
| Order_Cost | 619.81 | 1,054.82 | 1,606.19 | 986.38 | -859.76 | 3,085.76 |
| Tip_Amount | 23.50 | 46.48 | 69.60 | 46.10 | -45.65 | 138.75 |

Box plots for `Distance`, `Delivery_Person_Experience`, `Restaurant_Rating`, `Customer_Rating`, `Delivery_Time`, `Order_Cost`, and `Tip_Amount` were generated to visually confirm these outlier boundaries, alongside histogram + KDE distribution plots (with mean/median/mode overlaid) for each numeric column to assess skewness.

## 8. Visualizations & Key Insights

Across the Python notebook and the KNIME workflow (documented in full in `DAV Innovative Assignment.docx`/`.pdf`), the following plots and insights were produced:

- **Scatter — Delivery Time vs. Distance:** a broadly positive but non-linear relationship, with outliers (including physically impossible negative values in the raw data) needing careful handling before modeling.
- **Histogram — Delivery Time distribution:** most deliveries complete within roughly 40–120 minutes, with a tail at both ends and some invalid negative values in the raw data.
- **Histogram — Order Cost distribution:** most orders fall between 500–2000 cost units, with some low/negative values likely reflecting refunds or cancellations in the raw data.
- **Boxplot — Delivery Time by Traffic Conditions:** higher traffic is associated with longer and more variable delivery times, with outliers in both low- and high-traffic groups.
- **Boxplot — Delivery Time by Vehicle Type:** bikes, cars, and bicycles show similar median delivery times, but bike deliveries show higher variability.
- **Correlation heatmap (numeric features):** most feature pairs are weakly correlated; `Customer_Rating` shows a minor negative correlation with `Delivery_Time`, suggesting faster service may associate with better ratings.
- **Histogram — Tip Amount distribution:** tips cluster between roughly 20–100 units, with outliers on both ends.
- **Boxplot — Delivery Time by Weather Condition:** adverse weather (rain, snow) increases both the median and spread of delivery times; sunny/cloudy conditions correlate with quicker, more consistent deliveries.
- **IQR / outlier boundary table:** confirms significant spread and a non-trivial share of outliers across nearly every numeric column, underscoring the need for the preprocessing done in Section 6.
- **Barplot — Delivery Time by Order Priority:** higher-priority orders tend to have lower average delivery time, though variability remains high within every priority band.
- **Geographic scatter plot (Python):** customer and restaurant coordinates plotted together show deliveries spread broadly across the region with no single dominant cluster.
- **Hexbin density plot (Python):** reveals localized density clusters of delivery activity — potential hotspots for resource allocation.

> The project report notes that several location-based plots (geographic scatter, hexbin density) **could not be built natively in KNIME** and were instead generated in Python / Tableau, which is why the geographic analysis spans two tools.

## 9. KNIME Workflow

The `Knime/` folder contains a KNIME Analytics Platform (v5.7) workflow (`workflow.knime`), authored by Princy Pandya, that picks up after the Python cleaning step, reading **`Food_Delivery_Time_Prediction_custom_nulls.csv`** via a `CSV Reader` node and organizing the analysis into labeled sections (workflow annotations):

**Preprocessing & Visualization**
- `Missing Value` — imputation of remaining nulls
- `Rank Correlation`, `Heatmap` — correlation analysis
- `Bar Chart`, `Pie Chart`, `Density Plot`, `Histogram` (×2), `Line Plot` (×2), `Box Plot` (×4), `Scatter Plot` (×5) — the full EDA visual suite, mirrored/extended from the Python EDA
- `Normalizer` — feature scaling ahead of distance-based modeling

**K-Means Clustering**
- `X_Partitioner`, `k_Means` (configured for 3 clusters), `Cluster Assigner`, `Color Manager` — unsupervised segmentation of the delivery data

**Supervised learning experiments**, each following a **Table Partitioner → Learner → Predictor → Scorer** pattern:

| Model | Target column | Notes |
|---|---|---|
| **Decision Tree** | `Customer_Location` | Classification |
| **Naive Bayes** | `Weather_Conditions` | Classification; preceded by a `SMOTE` node to address class imbalance (input rebalanced to 690 rows) |
| **SVM** | `Weather_Conditions` | Classification (PMML-based SVM model) |
| **K-Nearest Neighbor** | `Weather_Conditions` | Classification, `k=3`, via the KNIME Distance Matrix extension (`Numeric Distances` + `K Nearest Neighbor`) |
| **Linear Regression** | `Tip_Amount` | Regression |
| **Random Forest (Regression)** | `Tip_Amount` | Regression |

Every learner feeds into a `Scorer` node for evaluation (accuracy/error for classifiers). The workflow file stores each Scorer's metric as a flow variable, but the actual computed accuracy/error values are held in KNIME's internal execution state rather than the exported XML, so they aren't reproduced here — **open the workflow in KNIME and re-execute it to see the live scores.**

> Note: the `k_Means` node's saved column selection still references an unrelated template's default columns (stock-price fields), which appear to be a leftover from the base KNIME example this node was built from rather than an intentional part of this project — worth double-checking/re-configuring if you re-run the workflow.

## 10. Tableau Dashboards

The repository includes Tableau workbook files (`InnovativeAssignment.twb`, `Dashboard1–3.twb`, `Sheet1–6.twb`) built on the **`Food_Delivery_Time_Location_Added_Only.csv`** data source (the cleaned dataset with parsed lat/long columns). `Dashboard1–3.twb` and `Sheet1–6.twb` are saved snapshots of the same underlying workbook at different points, containing six worksheets assembled into one dashboard ("Dashboard 1"):

| Sheet | Chart type | Purpose |
|---|---|---|
| Sheet 1 | Map (color by `Restaurant_Rating`) | Restaurant locations, colored by rating, filterable by traffic/weather and a "Passes Minimum Rating" calculation |
| Sheet 2 | Map | Restaurant locations filtered by average rating |
| Sheet 3 | Heatmap | Density of customer delivery locations |
| Sheet 4 | Chart | Order count vs. customer rating |
| Sheet 5 | Line chart | Order count by time of day (`Order_Time`) |
| Sheet 6 | Chart | Tip amount + average order cost by time of day |

**Interactive controls:**
- **Parameter — `Minimum Rating Threshold`** (range 1.0–5.0) feeding a calculated field **`Passes Minimum Rating`** = `[Restaurant_Rating] >= [Parameters].[Minimum Rating Threshold]`, letting users filter restaurants by a rating cutoff.
- **Parameter — `Delivery_Speed_Selection`** (`Fastest` / `Slowest`) feeding a calculated field **`Top_Bottom_Speed_Filter`** that uses `RANK()` on average `Delivery_Time` to isolate the top 10 fastest or bottom 10 slowest delivery groups.
- Several `Action` filters link `Order_ID` / `Order_Priority` / `Order_Time` selections across sheets for cross-filtering.

Together, the dashboard supports exploring **where** restaurants/customers are concentrated, **which** restaurants meet a minimum quality bar, **when** order volume peaks, and **which** conditions produce the fastest vs. slowest deliveries.

## 11. Results / Findings

- Delivery time is **not strongly linearly correlated** with any single numeric feature (all |r| < 0.11 against `Delivery_Time`); `Tip_Amount` and `Delivery_Person_Experience` show the (weak) strongest positive associations, `Restaurant_Rating` and `Distance` the (weak) strongest negative ones.
- **Weather** and **traffic** show a clearer categorical relationship with delivery time: adverse weather (rain/snow) and higher traffic are associated with longer and more variable delivery times.
- **Order priority** shows a mild association with faster average delivery, though with high within-group variability.
- **Vehicle type** shows similar median delivery times across bikes, cars, and bicycles, with bikes showing the most variability.
- The raw data contained a meaningful share of **outliers and invalid values** (e.g., negative distance, delivery time, cost, and tip values) that required explicit correction before analysis — a substantial part of the project's engineering effort.
- Given the weak linear correlations, the project's conclusion is that delivery time is likely driven by **non-linear, multi-feature interactions**, motivating the tree-based/ensemble and distance-based models built in the KNIME workflow rather than relying on a single linear model.
- Geographic analysis shows deliveries spread broadly across the region rather than concentrated in one hotspot, though the hexbin density plot does surface some localized clusters.

## 12. Project Structure

```
food-delivery-time-prediction-dav/
├── Innovative_Assignment.ipynb                          # Main notebook: cleaning + EDA (Python)
├── generated_null_values.ipynb                           # Injects synthetic nulls/invalid values into the base CSV
├── Food_Delivery_Time_Prediction.csv                      # Original base dataset (230 x 15)
├── Food_Delivery_Time_Prediction_custom_nulls.csv          # Base dataset + injected nulls (input to cleaning)
├── Food_Delivery_Time_Prediction_without_null_values.csv   # Cleaned dataset (217 x 15) — output of the notebook
├── Food_Delivery_Time_Location_Added_Only.csv              # Cleaned dataset + split lat/long columns (Tableau source)
├── Knime/                                                 # KNIME workflow (EDA, K-Means, Decision Tree, Naive Bayes,
│                                                           #   SVM, KNN, Linear Regression, Random Forest)
├── InnovativeAssignment.twb                               # Tableau workbook
├── Dashboard1.twb / Dashboard2.twb / Dashboard3.twb        # Tableau workbook snapshots (dashboard view)
├── Sheet1.twb … Sheet6.twb                                 # Tableau workbook snapshots (individual sheets)
├── DAV Innovative Assignment.docx / .pdf                   # Full written project report
└── README.md
```

## 13. How to Run / Reproduce the Project

**Python notebooks**
```bash
git clone https://github.com/princypandya/food-delivery-time-prediction-dav.git
cd food-delivery-time-prediction-dav
pip install pandas numpy scipy seaborn matplotlib jupyter
jupyter notebook Innovative_Assignment.ipynb
```
Run `generated_null_values.ipynb` first if you want to regenerate `Food_Delivery_Time_Prediction_custom_nulls.csv` from scratch (it reads `Food_Delivery_Time_Prediction.csv`); otherwise the file is already included in the repo. Then run `Innovative_Assignment.ipynb` top to bottom to reproduce the cleaning and EDA.

**KNIME workflow**
1. Install [KNIME Analytics Platform](https://www.knime.com/downloads) (workflow was authored on v5.7).
2. In KNIME, use *File → Import KNIME Workflow* and point it at the `Knime/` folder.
3. Update the `CSV Reader` node's file path if needed (it was originally authored against a local `D:\SEM-05\DAV\...` path) to point at `Food_Delivery_Time_Prediction_custom_nulls.csv` in your cloned repo.
4. Execute the workflow to reproduce the EDA visuals, K-Means clusters, and all six learner/predictor/scorer model chains.

**Tableau dashboards**
1. Install [Tableau Desktop](https://www.tableau.com/products/desktop) (or use Tableau Reader for view-only access).
2. Open `InnovativeAssignment.twb` (or any `Dashboard*.twb` file).
3. If prompted, repoint the data source to `Food_Delivery_Time_Location_Added_Only.csv` in your local copy of the repo.

## 14. Author

- **Princy Pandya** (23BCE207)
- **Yamini Pambhar** (23BCE204)

Institute of Technology, Nirma University — Semester V, Course 3CS103ME24 (Data Analysis and Visualization)
