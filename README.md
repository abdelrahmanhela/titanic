# Titanic Explorer (Streamlit)

Survival prediction for the Kaggle Titanic dataset — converted from the original
notebook into a full Python project, with a **test-set pipeline** that replays
every training transformation on `test.csv` and a Streamlit app to explore it.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Drop the Kaggle `train.csv` and `test.csv` into `Data/` — they load automatically.

## Test pipeline (CLI, no Streamlit needed)

```bash
python run_pipeline.py
python run_pipeline.py --model "Logistic Regression" --random-state 1
```

Writes `Data/submission.csv` (PassengerId, Survived).

## Structure

```
titanic/
├── app.py                  # entry point: st.navigation router
├── app_pages/
│   ├── overview.py         # data source, KPIs, previews, schema
│   ├── data_prep.py        # fitted imputation, class dummies, feature matrix
│   ├── eda.py              # survival distributions (lazy tabs)
│   ├── model.py            # sidebar training form, metrics, ROC, importance
│   └── predict.py          # test pipeline + submission download
├── utils/
│   ├── data.py             # load + fit/transform preprocessing (train-fitted)
│   ├── model.py            # training + evaluation
│   ├── pipeline.py         # test pipeline: transform → train → predict → submit
│   └── plots.py            # plotly figures
├── run_pipeline.py         # CLI entry point for the test pipeline
├── Data/                   # train.csv / test.csv (gitignored)
└── notebooks/              # original notebook (gitignored)
```

## The test pipeline

The notebook only transformed and scored the training data. `utils/pipeline.py`
closes that gap:

1. **Fit on train** — the Age-mean imputation value is learned from `train.csv`
   only (no leakage).
2. **Transform** — the same `build_features` function (imputation, Upper/Middle/
   Lower class dummies, Sex encoding, column order) is applied to both train and test.
3. **Train** — the model is fitted on the **full** training set.
4. **Predict** — `Survived` is predicted for every test passenger and packaged
   as a Kaggle submission.
