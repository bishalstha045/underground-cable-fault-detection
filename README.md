# Underground Cable Fault Detection System

This repository contains an Underground Cable Fault Detection System. It incorporates data generation, machine learning model training (using SVM, Random Forest, and Logistic Regression), and a fault identification/alert script.

## Project Structure

The project consists of the following key Python scripts:

*   **`generate_dataset.py`**: Generates a synthetic dataset (`underground_cable_dataset.csv`) of underground cable parameters including voltage, current, temperature, resistance, insulation resistance, and age. It also simulates faulty cables with corresponding parameter deviations.
*   **`train_model.py`**: Loads the generated dataset, preprocesses the data using `StandardScaler`, and trains three different machine learning models:
    *   Support Vector Machine (SVM)
    *   Random Forest Classifier
    *   Logistic Regression
    The script evaluates these models using accuracy, precision, recall, and F1-score. Finally, it saves the best performing model (`svm_model.pkl`) and the scaler (`scaler.pkl`) for future use.
*   **`faultidentify.py`**: Reads the dataset and functions as a monitoring/alerting system. It identifies cables marked as faulty and calculates a severity score (MODERATE, HIGH, CRITICAL) based on the deviation of current, temperature, resistance, and insulation resistance from normal thresholds.

## Required Libraries

To run the scripts in this repository, you will need the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn joblib
```

## How to Run

1.  **Generate Data**: Run the dataset generation script first to create the necessary CSV file.
    ```bash
    python generate_dataset.py
    ```
2.  **Train Models**: Train the machine learning models and save the best one.
    ```bash
    python train_model.py
    ```
3.  **Identify Faults**: Run the identification script to scan the cables and report any faults with their severity levels.
    ```bash
    python faultidentify.py
    ```

## Additional Files

*   **`underground_cable_dataset.csv`**: The generated dataset containing normal and faulty cable data.
*   **`svm_model.pkl` & `scaler.pkl`**: The saved Support Vector Machine model and its corresponding feature scaler.
*   **Various plots (`*.png`)**: Visualizations of model performance including confusion matrices, ROC curves, feature importance, and distributions.
*   **Research Papers (PDFs)**: Several reference research and survey papers related to the domain.
