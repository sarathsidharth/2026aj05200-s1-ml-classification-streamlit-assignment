# Term Deposit Subscription Classifier: Five ML Models + Streamlit App

**BITS Pilani · M.Tech (AIML) · Machine Learning, Assignment 2 · Student ID: 2026AJ05200**

This project trains five classification models on the UCI Bank Marketing dataset, compares them with six evaluation metrics, and serves them through an interactive Streamlit web app. The app loads the saved models and evaluates them on uploaded test data; it never retrains them.

---

## a. Problem statement

A Portuguese bank runs telephone marketing campaigns to sell **term deposits**. Most calls fail: only about 1 in 9 clients subscribes. So every call to an uninterested client costs agent time, and every subscriber who is never called is lost revenue.

**Objective:** build a **binary classifier** that predicts, *before* a client is called, whether that client will subscribe (`y = yes`) or not (`y = no`). The bank can then rank and prioritise its call list.

Because the classes are imbalanced (11.7% `yes`), a model that always answers "no" already reaches 88.3% accuracy while finding no subscribers at all. The models are therefore judged mainly on **MCC, F1 and ROC-AUC** for the positive class `yes`, not on accuracy alone.

## b. Dataset description

| Item | Details |
|---|---|
| **Name** | Bank Marketing, file `bank-full.csv` |
| **Source** | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing (Moro, Cortez & Rita, 2014) |
| **Size** | **45,211 instances**, 16 input features + 1 target (semicolon-separated); no missing (NaN) values and no duplicate rows |
| **Target** | `y`: has the client subscribed to a term deposit? `no` = 39,922 (88.3%), `yes` = 5,289 (11.7%). **Positive class = `yes`** (encoded 1) |
| **Features used** | **15** (assignment minimum: 12): 6 numerical + 9 categorical (`duration` excluded, see below) |
| **Train / test split** | Stratified 80:20, `random_state=42`: 36,168 training rows / **9,043 test rows**, both 11.70% `yes`. The test rows are saved as `test_data.csv`. |

**Features**

| Type | Feature | Meaning |
|---|---|---|
| Numerical | `age` | Client age (years) |
| | `balance` | Average yearly balance (EUR); heavily right-skewed, can be negative |
| | `day` | Day of the month of the last contact |
| | `campaign` | Number of contacts during this campaign |
| | `pdays` | Days since the client was last contacted in a previous campaign (-1 = never) |
| | `previous` | Number of contacts before this campaign |
| Categorical | `job` | 12 job types (admin., blue-collar, management, student, retired, …) |
| | `marital` | married / single / divorced |
| | `education` | primary / secondary / tertiary / unknown |
| | `default`, `housing`, `loan` | Has credit in default? Housing loan? Personal loan? (yes / no) |
| | `contact` | cellular / telephone / unknown |
| | `month` | Month of the last contact |
| | `poutcome` | Outcome of the previous campaign: success / failure / other / unknown |
| **Excluded** | `duration` | Length of the last call. It is only known **after** the call, when the outcome is also known, so using it would be **target leakage**. The UCI notes recommend discarding it for realistic prediction. |

**Preprocessing**, done inside each scikit-learn `Pipeline` and fitted on the **training split only**:

- **Numerical:** median imputation, then `StandardScaler`.
- **Categorical:** most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`. The output is dense, because GaussianNB requires it.
- `"unknown"` is kept as its own category. The EDA shows it is informative: `contact = unknown` subscribes at only 4.1%.
- Outliers in `balance`, `campaign`, `pdays` and `previous` are genuine values, so they are kept rather than removed.
- The 15 features become 50 model inputs.

**EDA highlights** (`notebooks/eda.ipynb`):
- `poutcome = success` subscribes at 64.7%, versus 11.7% overall.
- March, September, October and December convert at 44–52%; May converts at only 6.7%.
- Subscription by age is U-shaped: 42.6% for ages 66+, and about 9.4% for ages 36–55.
- `duration` correlates 0.395 with `y`, about 4× more than any other numerical feature. This is the leakage signal that we exclude.

## c. GitHub repository link

**Repository:** https://github.com/sarathsidharth/2026aj05200-s1-ml-classification-streamlit-assignment

```
2026aj05200-s1-ml-classification-streamlit-assignment/
├── app.py                  # Streamlit dashboard (inference only)
├── requirements.txt        # pinned dependencies (Python 3.12)
├── README.md
├── test_data.csv           # held-out 20% test split (9,043 rows, all original columns)
├── .gitignore
├── .streamlit/config.toml  # app theme (light + dark)
├── data/bank-full.csv      # UCI Bank Marketing dataset
├── model/                  # five saved preprocessing+classifier pipelines + metadata
│   ├── logistic_regression.pkl  decision_tree.pkl  knn.pkl  naive_bayes.pkl  svm.pkl
│   └── metadata.json       # split, parameters, train/test metrics, confusion matrices, versions
├── notebooks/eda.ipynb     # exploratory data analysis (executed, with outputs)
└── src/
    ├── __init__.py
    ├── preprocess.py       # feature definitions, CSV reader/validation, split, ColumnTransformer
    ├── train.py            # trains, evaluates and saves the five models
    └── evaluate.py         # six metrics, confusion matrix, report; re-checks saved models
```

## d. Models used

All five models use the **same dataset, the same 80:20 stratified split, the same preprocessing pipeline, and the same 9,043-row held-out test set**.

| Model | scikit-learn estimator and key hyperparameters | Why |
|---|---|---|
| Logistic Regression | `LogisticRegression(max_iter=1000, class_weight="balanced")` | Linear baseline; balanced weights counter the 88/12 imbalance |
| Decision Tree | `DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced")` | Non-linear rules; depth and leaf limits prevent overfitting |
| KNN | `KNeighborsClassifier(n_neighbors=15)` | Distance-based on scaled features; k=15 smooths noisy neighbourhoods (no class-weight option) |
| Naive Bayes | `GaussianNB()` | Fast probabilistic baseline (needs dense input) |
| Support Vector Machine | `SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced")` | Non-linear boundary; AUC from `decision_function` scores |

`random_state=42` is used wherever a model accepts it.

### Comparison table: held-out test set (n = 9,043, positive class = `yes`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | **0.6238** | 0.3732 | 0.2853 |
| Decision Tree | 0.8286 | 0.7792 | 0.3568 | 0.5794 | 0.4416 | 0.3613 |
| KNN | **0.8921** | 0.7349 | **0.6475** | 0.1701 | 0.2695 | 0.2939 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.8361 | **0.7914** | 0.3753 | 0.6030 | **0.4627** | **0.3867** |

Bold = best per metric.
- **Best AUC:** Support Vector Machine (0.7914).
- **Best F1:** Support Vector Machine (0.4627).
- **Best MCC:** Support Vector Machine (0.3867).
- **Best accuracy and precision:** KNN, with the lowest recall.
- **Best recall:** Logistic Regression.
- **Majority-class baseline** (always `no`): accuracy 0.8830, recall 0, MCC 0.

**Confusion matrices on the test set** (1,058 actual `yes`, 7,985 actual `no`):

| ML Model Name | TN | FP | FN | TP | Predicted `yes` |
|---|---|---|---|---|---|
| Logistic Regression | 6,166 | 1,819 | 398 | 660 | 2,479 |
| Decision Tree | 6,880 | 1,105 | 445 | 613 | 1,718 |
| KNN | 7,887 | 98 | 878 | 180 | 278 |
| Naive Bayes | 7,152 | 833 | 567 | 491 | 1,324 |
| Support Vector Machine | 6,923 | 1,062 | 420 | 638 | 1,700 |

**Training-split scores**, to check for overfitting (from `model/metadata.json`):

| ML Model Name | Train AUC | Test AUC | Train F1 | Test F1 | Train MCC | Test MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7681 | 0.7722 | 0.3768 | 0.3732 | 0.2900 | 0.2853 |
| Decision Tree | 0.7893 | 0.7792 | 0.4495 | 0.4416 | 0.3704 | 0.3613 |
| KNN | 0.8619 | 0.7349 | 0.3028 | 0.2695 | 0.3291 | 0.2939 |
| Naive Bayes | 0.7430 | 0.7514 | 0.3968 | 0.4123 | 0.3091 | 0.3271 |
| Support Vector Machine | 0.8474 | 0.7914 | 0.4995 | 0.4627 | 0.4313 | 0.3867 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Highest recall (0.6238)**: it finds 660 of the 1,058 subscribers, because `class_weight="balanced"` pushes the boundary towards `yes`. The price is the **lowest precision (0.2662)** and accuracy (0.7548): it flags 2,479 clients, and 1,819 of them are false alarms. It ranks clients reasonably (AUC 0.7722), but its **MCC is the lowest (0.2853)**. Train and test scores are almost identical (F1 0.3768 vs 0.3732), so it does not overfit. Its limit is the linear boundary, which cannot capture interactions such as month × previous outcome or the U-shaped age effect. |
| Decision Tree | **Second best overall**: MCC 0.3613, F1 0.4416 and AUC 0.7792, with recall 0.5794 at precision 0.3568. The depth limit (8) and minimum leaf size (20) keep overfitting small (train vs test AUC 0.7893 vs 0.7792). It captures non-linear rules (e.g. `poutcome = success`, high-converting months) and is easy to explain. Accuracy 0.8286 is below the 0.8830 baseline because the balanced weights favour recall. |
| KNN | **Highest accuracy (0.8921) and precision (0.6475), but recall is only 0.1701**: it finds just 180 of 1,058 subscribers and predicts `yes` only 278 times. Its accuracy is barely above the 0.8830 always-`no` baseline. KNN has no class-weight option, and in most 15-neighbour votes the majority class wins. It has the **lowest AUC (0.7349)**, the lowest F1 (0.2695), and the **largest overfitting gap** (train vs test AUC 0.8619 vs 0.7349), because it memorises training points. This model shows clearly why accuracy is misleading on this dataset. |
| Naive Bayes | **Middle of the pack**: MCC 0.3271, F1 0.4123, AUC 0.7514, with a balanced precision (0.3708) and recall (0.4641). It is the fastest to train (0.21 s) and does not overfit (test F1 0.4123 is above train F1 0.3968). Its independence and Gaussian assumptions do not fit the one-hot dummies and the skewed `balance`/`campaign` features, which limits its ranking quality (second-lowest AUC). |
| Support Vector Machine | **Best on all three imbalance-aware metrics: AUC 0.7914, F1 0.4627 and MCC 0.3867.** It finds 638 of 1,058 subscribers (recall 0.6030) while flagging 1,700 clients (precision 0.3753). Logistic Regression needs 2,479 flagged clients to find a similar number (660). The RBF kernel models non-linear patterns, and balanced class weights handle the imbalance. It overfits somewhat (train vs test AUC 0.8474 vs 0.7914) but is still best on unseen data. Costs: the slowest to train (about 71 s) and to score (about 10 s for 9,043 rows), and no calibrated probabilities (AUC uses `decision_function`). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF, balanced class weights).** It has the highest MCC (0.3867), F1 (0.4627) and AUC (0.7914), the three metrics that stay meaningful under an 88/12 imbalance. So it both ranks clients best and gives the best trade-off between finding subscribers and avoiding wasted calls. KNN has the highest accuracy (0.8921) but finds only 17% of subscribers, so accuracy is not used to pick the winner. **Decision Tree is a close runner-up** (MCC 0.3613), and is faster and more interpretable. The absolute scores (MCC about 0.39, AUC about 0.79) are moderate, because `duration` is excluded: they show what can realistically be predicted *before* a call is made. |

---

## Running the project

### 1. Setup (Python 3.12)

```bash
git clone https://github.com/sarathsidharth/2026aj05200-s1-ml-classification-streamlit-assignment.git
cd 2026aj05200-s1-ml-classification-streamlit-assignment
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`numpy 2.5.4` requires Python 3.12 or newer. On an older system Python, create the environment with Python 3.12, for example `uv venv --python 3.12` or `conda create -n ml2 python=3.12`.

### 2. Exploratory data analysis

```bash
pip install notebook                 # only needed for the notebook
jupyter notebook notebooks/eda.ipynb
```

### 3. Train the five models

```bash
python -m src.train
```

This writes `model/*.pkl`, `model/metadata.json` and `test_data.csv`, and prints the comparison table. It takes about 2 minutes, mostly for the SVM.

### 4. Reproduce and verify the results

```bash
python -m src.evaluate
```

This reloads the five saved pipelines, re-scores them on `test_data.csv`, and checks the six metrics against `model/metadata.json`. It exits with an error if anything differs. The split and all models use `random_state=42`, so retraining gives the same table.

### 5. Run the Streamlit app

```bash
streamlit run app.py                 # opens http://localhost:8501
```

**Using the app**
1. In the sidebar, either upload a labelled CSV (comma- or semicolon-separated, with the 15 features and `y`), or click **Use sample** / **Evaluate on the held-out test set**.
2. Choose one of the five models from the dropdown.
3. Read the results in the four tabs:
   - **Evaluation:** six metrics, confusion matrix and classification report.
   - **Model comparison:** leaderboard, metric chart and ROC curves.
   - **Predictions:** filter rows and download them as CSV.
   - **Dataset & models:** details of the data and the trained models.

The header button switches between light and dark mode.

The app only loads the saved models: uploaded data is transformed and scored, never used for fitting.

### 6. Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with GitHub and click **Create app**, then **Deploy a public app from GitHub**.
2. Set Repository to `sarathsidharth/2026aj05200-s1-ml-classification-streamlit-assignment`, Branch to `main`, and Main file path to `app.py`.
3. Under **Advanced settings**, select **Python 3.12**. Then click **Deploy**.

Streamlit Cloud installs `requirements.txt` and reads `.streamlit/config.toml`. No training happens at start-up, because the models are already in `model/`.

## Tech stack

Python 3.12 · scikit-learn 1.9.1 · pandas 3.1.0 · numpy 2.5.4 · Streamlit 1.66.1 · Altair 6.3.0 · matplotlib 3.11.2 · seaborn 0.13.2 · joblib 1.6.0
