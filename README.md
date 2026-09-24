# Car Price Predictor

A simple Streamlit web application that estimates the selling price of a used car from selected vehicle and technical details. The application uses a trained regression model and saved preprocessing files to generate the prediction.

## Project Overview

The project focuses on predicting used-car prices using machine learning regression. The training workflow prepares the dataset, handles categorical features with saved encoders, transforms the target price using a logarithm, compares regression models, and saves the selected model and preprocessing artifacts for use in the Streamlit application.

## Features

- Used-car price prediction through a Streamlit interface
- Brand, body type, engine type, registration status, and car model selection
- Numeric inputs for vehicle-related technical details
- Input validation against stored training ranges
- Saved model and preprocessing artifacts for direct prediction
- Model comparison using RMSE and R²
- Simple and lightweight local deployment with Streamlit

## Machine Learning

The project compares three regression models:

| Model | RMSE | R² |
|---|---:|---:|
| Linear Regression | 0.3880 | 0.8298 |
| Ridge Regression | 0.3880 | 0.8298 |
| Lasso Regression | 0.6542 | 0.5162 |

The target variable is `Price`. A `Log_price` transformation is used during training. Both `Price` and `Log_price` are excluded from the input features to avoid target leakage.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Project Structure

```text
Car-price-predictor/
│
├── app.py
├── train_model.py
├── car_price_model.pkl
├── encoders.pkl
├── cat_cols.pkl
├── feature_columns.pkl
├── numeric_ranges.pkl
├── model_results.csv
├── Dataset/
│   └── 1.04. Real-life example.csv
├── Linear_Ridge_&_Lasso_Regression_(2) (1).ipynb
├── requirements.txt
├── requirements-train.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

## Run the Application

1. Open the project folder in VS Code.
2. Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open at the local Streamlit address shown in the terminal.

## Retrain the Model

Retraining is optional because the trained model files are already included in the project.

To retrain the model:

```bash
pip install -r requirements-train.txt
python train_model.py
```

The training script reads the dataset, prepares the features, compares Linear, Ridge, and Lasso regression, evaluates them, and saves the model artifacts used by the application.

## Deployment on GitHub and Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the project files, including `app.py`, the saved `.pkl` files, `requirements.txt`, and the dataset/notebook files you want to keep in the repository.
3. Connect the repository to Streamlit Community Cloud.
4. Select `app.py` as the application entry point.
5. Keep `requirements.txt` in the repository so the required packages are installed.

## Model Artifacts

The Streamlit application loads these saved files at runtime:

- `car_price_model.pkl` — trained regression model
- `encoders.pkl` — encoders for categorical features
- `cat_cols.pkl` — categorical feature list
- `feature_columns.pkl` — feature order expected by the model
- `numeric_ranges.pkl` — stored valid input ranges

## Notes

The displayed price is a machine-learning estimate and may differ from an actual market selling price. The current application converts the model output to Indian Rupees using the conversion used in the project code.

## Author & Contact

- **Name:** Mohammad Somama
- **Role:** Data Analyst
- **Email:** [mohammadsomama01@gmail.com](mailto:mohammadsomama01@gmail.com)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/mohammad-somama-67a825315/)
- **GitHub:** [Mohammad Somama](https://github.com/mohammad-somama)
