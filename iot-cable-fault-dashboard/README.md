# IoT Underground Cable Fault Detection Dashboard

This folder contains a Streamlit dashboard for simulated real-time IoT monitoring and machine learning-based fault detection.

## Files

- `app.py` - Streamlit dashboard application
- `requirements.txt` - Python dependencies

## Run Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

3. In the browser, click **Start Monitoring** to begin the live IoT simulation.

## Notes

- The app loads `svm_model.pkl` and `scaler.pkl` from the workspace root.
- It simulates voltage, current, temperature, resistance, insulation resistance, cable age, and cable length.
- Fault injection is enabled for frequent demo-ready alerts.
