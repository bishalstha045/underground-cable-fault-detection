from pathlib import Path
import random
import time
import os
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATASET_PATH = BASE_DIR / "underground_cable_dataset.csv"

FEATURE_COLUMNS = [
    "voltage",
    "current",
    "temperature",
    "resistance",
    "insulation_resistance",
    "cable_age",
    "cable_length_km",
    "fault_distance_km",
]

st.set_page_config(
    page_title="IoT Underground Cable Fault Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df.reset_index(drop=True)
    df.insert(0, "cable_id", df.index + 1)
    return df


def trigger_rerun():
    rerun_fn = getattr(st, "rerun", None)
    if rerun_fn is not None:
        rerun_fn()


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def simulate_values(values):
    next_values = values.copy()
    next_values["voltage"] = clamp(values["voltage"] + random.uniform(-2.8, 3.2), 215, 250)
    next_values["current"] = clamp(values["current"] + random.uniform(-1.6, 1.6), 9.0, 36.5)
    next_values["temperature"] = clamp(values["temperature"] + random.uniform(-1.2, 1.8), 25.0, 58.0)
    next_values["resistance"] = clamp(values["resistance"] + random.uniform(-0.18, 0.18), 0.7, 4.5)
    next_values["insulation_resistance"] = clamp(values["insulation_resistance"] + random.uniform(-4.2, 4.5), 30.0, 124.0)
    next_values["cable_age"] = clamp(values["cable_age"] + random.uniform(-0.05, 0.05), 1.0, 22.0)
    next_values["cable_length"] = clamp(values["cable_length"] + random.uniform(-5, 5), 80.0, 1250.0)
    next_values["fault_distance_km"] = clamp(
        next_values["cable_length"] * random.uniform(0.03, 0.55) / 1000.0,
        0.01,
        next_values["cable_length"] / 1000.0,
    )

    fault_injected = False
    if random.random() < 0.36:
        fault_injected = True
        next_values["temperature"] = clamp(next_values["temperature"] + random.uniform(4.5, 14.0), 35.0, 72.0)
        next_values["resistance"] = clamp(next_values["resistance"] + random.uniform(0.5, 1.8), 1.6, 6.0)
        next_values["insulation_resistance"] = clamp(next_values["insulation_resistance"] - random.uniform(16.0, 40.0), 18.0, 95.0)
        next_values["current"] = clamp(next_values["current"] + random.uniform(1.4, 6.6), 10.0, 42.5)
        next_values["voltage"] = clamp(next_values["voltage"] - random.uniform(2.5, 8.5), 205.0, 245.0)
        next_values["fault_distance_km"] = clamp(
            next_values["cable_length"] * random.uniform(0.03, 0.35) / 1000.0,
            0.01,
            next_values["cable_length"] / 1000.0,
        )

    return next_values, fault_injected


def compute_severity(prob, row):
    thermal_score = min(1.0, row["temperature"] / 65.0)
    current_score = min(1.0, row["current"] / 70.0)
    resistive_score = min(1.0, row["resistance"] / 0.8)
    insulation_score = min(1.0, max(0.0, (2.0 - row["insulation_resistance"]) / 1.7))
    stress_index = (0.40 * prob) + (0.16 * thermal_score) + (0.16 * current_score) + (0.14 * resistive_score) + (0.14 * insulation_score)

    if stress_index > 0.75:
        return "CRITICAL"
    if stress_index > 0.50:
        return "HIGH"
    return "MODERATE"


def classify_fault_type(row):
    if row["insulation_resistance"] < 0.9 and row["resistance"] > 0.58:
        return "Insulation Breakdown"
    if row["current"] > 55 and row["temperature"] > 48:
        return "Thermal Overload"
    if row["voltage"] < 9.0 and row["resistance"] > 0.55:
        return "Conductor Degradation"
    if row["fault_distance_km"] > 0.0 and row["temperature"] > 42:
        return "Localized Cable Fault"
    return "Mixed Electrical Stress"


def normalize_phone(raw_value):
    return "".join(ch for ch in str(raw_value).strip() if ch in "+0123456789")


def parse_recipient_numbers(raw_value):
    parts = [p.strip() for p in str(raw_value).split(",") if p.strip()]
    return [normalize_phone(p) for p in parts if normalize_phone(p)]


def choose_recipients_by_fault_distance(values, recipients):
    if not recipients:
        return [], "NO_RECIPIENT"
    if len(recipients) == 1:
        return [recipients[0]], "ONLY_ONE_RECIPIENT"

    fault_distance_km = float(values.get("fault_distance_km", 0.0))
    if fault_distance_km < 0.85:
        return [recipients[0]], "LT_0_85KM_TO_RECIPIENT_1"

    return [recipients[1]], "GTE_0_85KM_TO_RECIPIENT_2"


def format_twilio_error(response):
    try:
        payload = response.json()
        code = payload.get("code")
        message = payload.get("message")
        return f"HTTP {response.status_code} | Twilio {code}: {message}"
    except Exception:
        text = (response.text or "").replace("\n", " ")
        return f"HTTP {response.status_code} | {text[:220]}"


def twilio_error_hint(response):
    try:
        payload = response.json()
        code = payload.get("code")
    except Exception:
        code = None

    if response.status_code == 401 or code == 20003:
        return (
            "Fix: Use Account SID (starts with AC) and Auth Token from the same Twilio account/subaccount. "
            "Do not use API Key SID (SK...) here."
        )
    if code == 63007:
        return "Fix: WhatsApp From is not enabled for this account/channel. Use sandbox From (+14155238886) or an approved WhatsApp sender."
    if code == 63016:
        return "Fix: Recipient likely has not joined sandbox or message is outside allowed window. Re-join sandbox or use approved template."
    if code == 63038:
        return "Fix: Twilio daily sandbox/business-initiated limit reached. Wait for reset or upgrade sender setup."
    return "Fix: Verify From/To format (+countrycode...), sandbox join, and template fields when template mode is enabled."


def sanitize_secret_text(value):
    if value is None:
        return ""
    cleaned = str(value).strip().strip('"').strip("'")
    # Remove all whitespace characters commonly introduced by copy/paste.
    cleaned = "".join(cleaned.split())
    return cleaned


def get_twilio_credentials():
    account_sid = sanitize_secret_text(st.session_state.get("twilio_sid", ""))
    auth_token = sanitize_secret_text(st.session_state.get("twilio_token", ""))

    if not account_sid or not auth_token:
        return None, None, "Twilio SID/Token missing."

    if not account_sid.startswith("AC") or len(account_sid) != 34:
        return None, None, (
            "Twilio SID format looks invalid. It must be your Account SID "
            "starting with AC and 34 characters long."
        )

    if len(auth_token) < 20:
        return None, None, (
            "Twilio Auth Token looks too short. Paste the full token from Twilio Console."
        )

    return account_sid, auth_token, ""


def ensure_live_twilio_auth(account_sid, auth_token):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json"
    try:
        resp = requests.get(url, auth=(account_sid, auth_token), timeout=8)
    except Exception as exc:
        return f"Twilio auth precheck failed (network): {exc}"

    if resp.status_code >= 400:
        return f"Twilio auth precheck failed: {format_twilio_error(resp)}. {twilio_error_hint(resp)}"
    return ""


def validate_twilio_credentials():
    account_sid, auth_token, err = get_twilio_credentials()
    if err:
        st.session_state["notify_status"] = err
        return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json"
    resp = requests.get(url, auth=(account_sid, auth_token), timeout=8)
    if resp.status_code >= 400:
        st.session_state["notify_status"] = (
            f"Credential validation failed: {format_twilio_error(resp)}. {twilio_error_hint(resp)}"
        )
        return
    st.session_state["notify_status"] = "Twilio credentials validated successfully."


def build_twilio_payload(body_text, severity=None, fault_type=None, cable_id=None):
    from_whatsapp = normalize_phone(st.session_state.get("from_whatsapp", ""))
    payload = {
        "From": f"whatsapp:{from_whatsapp}",
    }

    use_template = st.session_state.get("use_template_mode", False)
    content_sid = st.session_state.get("twilio_content_sid", "").strip()
    content_variables = st.session_state.get("twilio_content_variables", "").strip()

    if use_template and content_sid:
        payload["ContentSid"] = content_sid
        if content_variables:
            payload["ContentVariables"] = content_variables
        else:
            auto_vars = {
                "1": str(cable_id or "N/A"),
                "2": str(severity or "N/A"),
                "3": str(fault_type or "N/A"),
            }
            payload["ContentVariables"] = json.dumps(auto_vars)
    else:
        payload["Body"] = body_text

    return payload


def send_fault_notifications(severity, fault_type, values):
    if not st.session_state.get("notify_enabled", False):
        return

    account_sid, auth_token, err = get_twilio_credentials()
    from_whatsapp = normalize_phone(st.session_state.get("from_whatsapp", ""))
    to_whatsapp_list = parse_recipient_numbers(st.session_state.get("to_whatsapp", ""))
    from_call = normalize_phone(st.session_state.get("from_call", ""))
    to_call = normalize_phone(st.session_state.get("to_call", ""))

    if err:
        st.session_state["notify_status"] = err
        return

    auth_err = ensure_live_twilio_auth(account_sid, auth_token)
    if auth_err:
        st.session_state["notify_status"] = auth_err
        return

    if not from_whatsapp or not to_whatsapp_list:
        return

    routed_recipients, route_label = choose_recipients_by_fault_distance(values, to_whatsapp_list)
    if not routed_recipients:
        st.session_state["notify_status"] = "No valid recipient available for distance-based routing."
        return

    notification_key = f"{values['cable_id']}-{severity}-{fault_type}"

    message_body = (
        f"Fault detected ({severity})\n"
        f"Cable ID: {values['cable_id']}\n"
        f"Fault Type: {fault_type}\n"
        f"Voltage: {values['voltage']:.3f} V\n"
        f"Current: {values['current']:.3f} A\n"
        f"Temperature: {values['temperature']:.3f} C\n"
        f"Resistance: {values['resistance']:.3f} ohm\n"
        f"Insulation: {values['insulation_resistance']:.3f} MOhm\n"
        f"Cable Age: {values['cable_age']:.3f} years\n"
        f"Cable Length: {values['cable_length'] / 1000.0:.3f} km\n"
        f"Fault Distance: {values['fault_distance_km']:.3f} km"
    )

    base_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"
    auth = (account_sid, auth_token)

    try:
        sent_count = 0
        failed_recipients = []
        per_recipient_results = []
        first_sid = ""
        first_status = ""

        for to_whatsapp in routed_recipients:
            msg_payload = build_twilio_payload(
                body_text=message_body,
                severity=severity,
                fault_type=fault_type,
                cable_id=values.get("cable_id"),
            )
            msg_payload["To"] = f"whatsapp:{to_whatsapp}"

            msg_resp = requests.post(
                f"{base_url}/Messages.json",
                data=msg_payload,
                auth=auth,
                timeout=8,
            )

            if msg_resp.status_code >= 400:
                err_text = f"{format_twilio_error(msg_resp)}. {twilio_error_hint(msg_resp)}"
                failed_recipients.append(f"{to_whatsapp} ({msg_resp.status_code}) {err_text}")
                per_recipient_results.append(f"{to_whatsapp}: FAILED {err_text}")
                continue

            msg_json = msg_resp.json()
            msg_sid = msg_json.get("sid", "")
            msg_status = msg_json.get("status", "accepted")
            if not first_sid:
                first_sid = msg_sid
                first_status = msg_status
            st.session_state["last_message_sid"] = msg_sid
            sent_count += 1
            per_recipient_results.append(f"{to_whatsapp}: SENT SID={msg_sid} STATUS={msg_status}")

        if sent_count == 0:
            first_error = per_recipient_results[0] if per_recipient_results else "Unknown send failure"
            st.session_state["notify_status"] = (
                "WhatsApp send failed for all recipients. "
                f"First error: {first_error}. "
                "Check sandbox join, verified numbers, Twilio WhatsApp From, and ContentSid/ContentVariables (if template mode enabled)."
            )
            st.session_state["alert_log"].extend(per_recipient_results)
            return

        if from_call and to_call:
            call_twiml = (
                f"<Response><Say voice='alice'>Fault detected. Severity {severity}. Cable ID {values['cable_id']}. "
                f"Fault type {fault_type}. Please check dashboard immediately.</Say></Response>"
            )
            requests.post(
                f"{base_url}/Calls.json",
                data={
                    "From": from_call,
                    "To": to_call,
                    "Twiml": call_twiml,
                },
                auth=auth,
                timeout=8,
            )

        if failed_recipients:
            st.session_state["notify_status"] = (
                f"WhatsApp alert sent to {sent_count}/{len(routed_recipients)} routed recipients ({route_label}). "
                f"First SID: {first_sid} Status: {first_status}. Failed: {', '.join(failed_recipients)}"
            )
        else:
            st.session_state["notify_status"] = (
                f"WhatsApp alert accepted by Twilio for {sent_count} routed recipients ({route_label}). "
                f"First SID: {first_sid} Status: {first_status}"
            )
        st.session_state["last_notified_key"] = notification_key
        st.session_state["alert_log"].append(
            f"Cable {values['cable_id']}: ROUTE={route_label} SENT to {sent_count}/{len(routed_recipients)} recipients ({severity}, {fault_type})"
        )
        st.session_state["alert_log"].extend(per_recipient_results)
    except Exception as exc:
        st.session_state["notify_status"] = f"Notification error: {exc}"
        st.session_state["alert_log"].append(
            f"Cable {values['cable_id']}: FAILED ({exc})"
        )


def send_test_whatsapp_message():
    account_sid, auth_token, err = get_twilio_credentials()
    from_whatsapp = normalize_phone(st.session_state.get("from_whatsapp", ""))
    to_whatsapp_list = parse_recipient_numbers(st.session_state.get("to_whatsapp", ""))

    if err:
        st.session_state["notify_status"] = err
        return

    auth_err = ensure_live_twilio_auth(account_sid, auth_token)
    if auth_err:
        st.session_state["notify_status"] = auth_err
        return

    if not from_whatsapp or not to_whatsapp_list:
        st.session_state["notify_status"] = "Missing WhatsApp From/To numbers for test message."
        return

    message_body = (
        "Test alert from IoT Underground Cable Fault Dashboard. "
        "If you received this, Twilio WhatsApp integration is working."
    )

    base_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"
    auth = (account_sid, auth_token)
    try:
        sent_count = 0
        failed_recipients = []
        per_recipient_results = []
        first_sid = ""
        first_status = ""

        for to_whatsapp in to_whatsapp_list:
            msg_payload = build_twilio_payload(
                body_text=message_body,
                severity="TEST",
                fault_type="Template Test",
                cable_id="TEST",
            )
            msg_payload["To"] = f"whatsapp:{to_whatsapp}"
            msg_resp = requests.post(
                f"{base_url}/Messages.json",
                data=msg_payload,
                auth=auth,
                timeout=10,
            )
            if msg_resp.status_code >= 400:
                err_text = f"{format_twilio_error(msg_resp)}. {twilio_error_hint(msg_resp)}"
                failed_recipients.append(f"{to_whatsapp} ({msg_resp.status_code}) {err_text}")
                per_recipient_results.append(f"{to_whatsapp}: FAILED {err_text}")
                continue

            msg_json = msg_resp.json()
            msg_sid = msg_json.get("sid", "")
            msg_status = msg_json.get("status", "accepted")
            if not first_sid:
                first_sid = msg_sid
                first_status = msg_status
            st.session_state["last_message_sid"] = msg_sid
            sent_count += 1
            per_recipient_results.append(f"{to_whatsapp}: SENT SID={msg_sid} STATUS={msg_status}")

        if sent_count == 0:
            first_error = per_recipient_results[0] if per_recipient_results else "Unknown send failure"
            st.session_state["notify_status"] = (
                "Test message failed for all recipients. "
                f"First error: {first_error}. "
                "Most common fix: join Twilio WhatsApp sandbox from each recipient phone first, "
                "or configure ContentSid/ContentVariables correctly in template mode."
            )
            st.session_state["alert_log"].extend(per_recipient_results)
            return

        if failed_recipients:
            st.session_state["notify_status"] = (
                f"Test WhatsApp sent to {sent_count}/{len(to_whatsapp_list)} recipients. "
                f"First SID: {first_sid} Status: {first_status}. Failed: {', '.join(failed_recipients)}"
            )
        else:
            st.session_state["notify_status"] = (
                f"Test WhatsApp accepted by Twilio for {sent_count} recipients. "
                f"First SID: {first_sid} Status: {first_status}"
            )
        st.session_state["alert_log"].append(
            f"TEST: SENT to {sent_count}/{len(to_whatsapp_list)} recipients"
        )
        st.session_state["alert_log"].extend(per_recipient_results)
    except Exception as exc:
        st.session_state["notify_status"] = f"Test message error: {exc}"


def fetch_last_message_status():
    account_sid, auth_token, err = get_twilio_credentials()
    msg_sid = st.session_state.get("last_message_sid", "")

    if err:
        st.session_state["notify_status"] = err
        return

    if not msg_sid:
        st.session_state["notify_status"] = "No recent Twilio message SID available to check status."
        return

    base_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"
    auth = (account_sid, auth_token)

    try:
        resp = requests.get(f"{base_url}/Messages/{msg_sid}.json", auth=auth, timeout=8)
        if resp.status_code >= 400:
            st.session_state["notify_status"] = f"Status check failed ({resp.status_code}): {resp.text}"
            return
        msg = resp.json()
        status = msg.get("status", "unknown")
        error_code = msg.get("error_code")
        error_message = msg.get("error_message")
        st.session_state["notify_status"] = (
            f"Twilio delivery status for SID {msg_sid}: {status}. "
            f"Error code: {error_code}. Error message: {error_message}"
        )
        st.session_state["alert_log"].append(
            f"STATUS SID={msg_sid} STATUS={status} ERR={error_code}"
        )
    except Exception as exc:
        st.session_state["notify_status"] = f"Status check error: {exc}"


def init_session_state():
    if "monitoring" not in st.session_state:
        st.session_state["monitoring"] = False
    if "values" not in st.session_state:
        st.session_state["values"] = {
            "cable_id": 0,
            "voltage": 230.0,
            "current": 14.0,
            "temperature": 36.5,
            "resistance": 1.4,
            "insulation_resistance": 92.0,
            "cable_age": 8.5,
            "cable_length": 420.0,
            "fault_distance_km": 0.25,
        }
        st.session_state["previous"] = st.session_state["values"].copy()
        st.session_state["history"] = pd.DataFrame([st.session_state["values"]])
        st.session_state["alert_state"] = "No Fault"
        st.session_state["iteration"] = 0

    # Always ensure keys exist, even for old/stale Streamlit sessions.
    if "previous" not in st.session_state:
        st.session_state["previous"] = st.session_state["values"].copy()
    if "history" not in st.session_state:
        st.session_state["history"] = pd.DataFrame([st.session_state["values"]])
    if "alert_state" not in st.session_state:
        st.session_state["alert_state"] = "No Fault"
    if "iteration" not in st.session_state:
        st.session_state["iteration"] = 0
    if "demo_index" not in st.session_state:
        st.session_state["demo_index"] = 0
    if "last_notified_key" not in st.session_state:
        st.session_state["last_notified_key"] = None
    if "notify_status" not in st.session_state:
        st.session_state["notify_status"] = ""
    if "seen_fault_ids" not in st.session_state:
        st.session_state["seen_fault_ids"] = set()
    if "seen_fault_keys" not in st.session_state:
        st.session_state["seen_fault_keys"] = set()
    if "alert_log" not in st.session_state:
        st.session_state["alert_log"] = []
    if "last_message_sid" not in st.session_state:
        st.session_state["last_message_sid"] = ""
    if "use_template_mode" not in st.session_state:
        st.session_state["use_template_mode"] = False
    if "twilio_content_sid" not in st.session_state:
        st.session_state["twilio_content_sid"] = ""
    if "twilio_content_variables" not in st.session_state:
        st.session_state["twilio_content_variables"] = ""
        st.session_state["seen_fault_ids"] = set()
        st.session_state["seen_fault_keys"] = set()
        st.session_state["alert_log"] = []


def render_dashboard(model, scaler, dataset):
    st.markdown("## ⚡ IoT Underground Cable Fault Detection Dashboard")
    st.markdown("#### Live streaming, ML inference, and fault alert monitoring")
    st.caption("Real-time SCADA-style monitor with route-based fault notifications and predictive ML analysis.")

    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.caption("SCADA-style monitoring panel with simulated IoT stream")
    with head_right:
        if st.session_state["monitoring"]:
            st.markdown("<div class='status-chip' style='background:#14b8a6;color:#0f172a;'>LIVE STREAMING</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-chip' style='background:#f59e0b;color:#1f2937;'>PAUSED</div>", unsafe_allow_html=True)

    control_col, _ = st.columns([1, 3])
    with control_col:
        toggle_label = "⏹ Stop Monitoring" if st.session_state["monitoring"] else "▶ Start Monitoring"
        if st.button(toggle_label, use_container_width=True):
            st.session_state["monitoring"] = not st.session_state["monitoring"]
            trigger_rerun()

    st.markdown("---")
    live_label = st.empty()
    progress_holder = st.empty()

    st.markdown("### 🧭 LIVE SENSOR PANEL")
    sensor_grid = st.container()
    with sensor_grid:
        columns = st.columns(4)
        current = st.session_state["values"]
        previous = st.session_state["previous"]
        metric_defs = [
            ("Cable ID", current.get("cable_id", 0), current.get("cable_id", 0) - previous.get("cable_id", 0), False),
            ("Voltage (V)", current["voltage"], current["voltage"] - previous["voltage"], current["voltage"] > 240 or current["voltage"] < 218),
            ("Current (A)", current["current"], current["current"] - previous["current"], current["current"] > 31.5 or current["current"] < 10.5),
            ("Temperature (deg C)", current["temperature"], current["temperature"] - previous["temperature"], current["temperature"] > 50.0),
            ("Resistance (ohm)", current["resistance"], current["resistance"] - previous["resistance"], current["resistance"] > 3.4),
            ("Insulation (MOhm)", current["insulation_resistance"], current["insulation_resistance"] - previous["insulation_resistance"], current["insulation_resistance"] < 60),
            ("Cable Age (yrs)", current["cable_age"], current["cable_age"] - previous["cable_age"], False),
            ("Length (m)", current["cable_length"], current["cable_length"] - previous["cable_length"], False),
            ("Fault Dist (km)", current["fault_distance_km"], current["fault_distance_km"] - previous["fault_distance_km"], False),
        ]
        for idx, metric in enumerate(metric_defs):
            columns[idx % 4].metric(metric[0], f"{metric[1]:.2f}", f"{metric[2]:+.2f}", delta_color="inverse" if metric[3] else "normal")

    st.markdown("### 📈 REAL-TIME GRAPHS")
    graph_cols = st.columns(2)
    graph_cols[0].markdown("#### Temperature vs Time")
    graph_cols[1].markdown("#### Current vs Time")
    temp_chart = graph_cols[0].empty()
    current_chart = graph_cols[1].empty()
    st.markdown("#### Temperature vs Current")
    scatter_chart = st.empty()

    st.markdown("### 🧠 ML ANALYSIS PANEL")
    ml_cols = st.columns(3)
    ml_status = ml_cols[0].empty()
    ml_prob = ml_cols[1].empty()
    ml_details = ml_cols[2].empty()

    st.markdown("### 🚨 VISUAL ALERT SYSTEM")
    alert_placeholder = st.empty()

    if st.session_state["monitoring"]:
        idx = st.session_state["demo_index"] % len(dataset)
        row_data = dataset.iloc[idx]
        st.session_state["demo_index"] += 1

        new_values = {
            "cable_id": int(row_data["cable_id"]),
            "voltage": float(row_data["voltage"]),
            "current": float(row_data["current"]),
            "temperature": float(row_data["temperature"]),
            "resistance": float(row_data["resistance"]),
            "insulation_resistance": float(row_data["insulation_resistance"]),
            "cable_age": float(row_data["cable_age"]),
            "cable_length": float(row_data["cable_length_km"] * 1000.0),
            "fault_distance_km": float(row_data["fault_distance_km"]),
        }

        st.session_state["previous"] = st.session_state["values"].copy()
        st.session_state["values"] = new_values
        st.session_state["iteration"] += 1

        row = pd.DataFrame([new_values])
        st.session_state["history"] = pd.concat([st.session_state["history"], row], ignore_index=True).tail(30)
        sensor_df = st.session_state["history"][["temperature", "current"]].reset_index(drop=True)
        temp_chart.line_chart(sensor_df["temperature"], height=280, use_container_width=True)
        current_chart.line_chart(sensor_df["current"], height=280, use_container_width=True)
        scatter_chart.scatter_chart(sensor_df.rename(columns={"temperature": "Temperature", "current": "Current"}), x="Temperature", y="Current", height=260, use_container_width=True)

        features = np.array(
            [[
                new_values["voltage"],
                new_values["current"],
                new_values["temperature"],
                new_values["resistance"],
                new_values["insulation_resistance"],
                new_values["cable_age"],
                new_values["cable_length"] / 1000.0,
                new_values["fault_distance_km"],
            ]]
        )

        scaled = scaler.transform(features)
        try:
            probability = float(model.predict_proba(scaled)[0][1])
        except Exception:
            probability = float(model.decision_function(scaled)[0])
            probability = 1 / (1 + np.exp(-probability))

        predicted = int(model.predict(scaled)[0])
        dataset_label_fault = int(row_data["label"]) == 1
        fault_status = "FAULT DETECTED" if (predicted == 1 or dataset_label_fault) else "NO FAULT"
        fault_type = classify_fault_type(new_values) if fault_status == "FAULT DETECTED" else "Healthy"
        severity = compute_severity(probability, new_values)
        severity_color = "#ef4444" if severity == "CRITICAL" else "#f59e0b" if severity == "HIGH" else "#22c55e"
        status_color = "#dc2626" if fault_status == "FAULT DETECTED" else "#16a34a"
        fault_distance_m = new_values["fault_distance_km"] * 1000.0

        ml_status.markdown(
            f"<div class='card' style='background:#ffffff;border:1px solid #d1d5db;border-radius:12px;padding:14px;'><h4 style='color:#111827;font-weight:800;'>Fault Status</h4><p style='font-size:2rem;font-weight:900;color:{status_color};'>{fault_status}</p></div>",
            unsafe_allow_html=True,
        )
        ml_prob.markdown(
            f"<div class='card' style='background:#ffffff;border:1px solid #d1d5db;border-radius:12px;padding:14px;'><h4 style='color:#111827;font-weight:800;'>Probability</h4><p style='font-size:2rem;font-weight:900;color:#1d4ed8;'>{probability*100:.1f}%</p></div>",
            unsafe_allow_html=True,
        )
        ml_details.markdown(
            f"<div class='card' style='background:#ffffff;border:1px solid #d1d5db;border-radius:12px;padding:14px;'><h4 style='color:#111827;font-weight:800;'>Fault Distance</h4><p style='font-size:1.9rem;font-weight:900;color:#111827;'>{fault_distance_m:.1f} m</p><hr style='border:1px solid #d1d5db; margin:12px 0;'/><div style='font-size:1.05rem;color:#111827;font-weight:700;'>Severity: <span style='color:{severity_color}; font-weight:900;'>{severity}</span></div><div style='font-size:1.05rem;color:#111827;font-weight:700;'>Fault Type: <span style='color:#dc2626; font-weight:900;'>{fault_type}</span></div></div>",
            unsafe_allow_html=True,
        )

        if fault_status == "FAULT DETECTED":
            alert_placeholder.error("🛑 ALERT: Fault detected in cable network. Immediate attention required.", icon="🚨")
            st.markdown("<p class='blink'>⚠️ Flashing Alert: Fault signature detected!</p>", unsafe_allow_html=True)
            recipients = parse_recipient_numbers(st.session_state.get("to_whatsapp", ""))
            _, route_label = choose_recipients_by_fault_distance(new_values, recipients)
            fault_event_key = f"{new_values['cable_id']}:{route_label}"
            if dataset_label_fault and fault_event_key not in st.session_state["seen_fault_keys"]:
                send_fault_notifications(severity, fault_type, new_values)
                st.session_state["seen_fault_keys"].add(fault_event_key)
        else:
            alert_placeholder.success("✅ System stable. No faults currently detected.", icon="✅")

        progress_holder.progress(min(100, 40 + int(probability * 60)))
        live_label.markdown(
            f"<div style='font-size:0.95rem;color:#c7d2fe;'>📡 All-cable stream... update #{st.session_state['iteration']} | Cable ID {new_values['cable_id']}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(2)
        trigger_rerun()
    else:
        sensor_df = st.session_state["history"][["temperature", "current"]].reset_index(drop=True)
        if not sensor_df.empty:
            temp_chart.line_chart(sensor_df["temperature"], height=280, use_container_width=True)
            current_chart.line_chart(sensor_df["current"], height=280, use_container_width=True)
            scatter_chart.scatter_chart(sensor_df.rename(columns={"temperature": "Temperature", "current": "Current"}), x="Temperature", y="Current", height=260, use_container_width=True)
        ml_status.markdown("<div class='card' style='background:#ffffff;border:1px solid #d1d5db;border-radius:12px;padding:14px;'><p style='color:#111827;font-weight:700;'>Press ▶ Start Monitoring to begin the live IoT dashboard.</p></div>", unsafe_allow_html=True)
        ml_prob.empty()
        ml_details.empty()
        alert_placeholder.info("🔷 Dashboard is paused. Click Start Monitoring to simulate live sensor flow.")
        progress_holder.progress(0)
        live_label.markdown("<div style='font-size:0.95rem;color:#a5b4fc;'>▶ Awaiting live stream...</div>", unsafe_allow_html=True)


def render_safe_page(dataset: pd.DataFrame):
    st.markdown("## ✅ Safe Cables")
    safe_df = dataset[dataset["label"] == 0].copy()
    st.info(f"Dataset rows analyzed: {len(dataset)}")
    st.success(f"Safe cables found: {len(safe_df)}")

    st.markdown("### Safe Cable IDs")
    id_text = ", ".join(str(cid) for cid in safe_df["cable_id"].tolist())
    st.write(id_text if id_text else "No safe cables found.")

    st.markdown("### Safe Cable Table")
    st.dataframe(safe_df[["cable_id"] + FEATURE_COLUMNS], use_container_width=True)


def plot_fault_image(fault_row: pd.Series):
    labels = [
        "voltage",
        "current",
        "temperature",
        "resistance",
        "insulation",
        "age",
        "length_km",
        "distance_km",
    ]
    values = [
        fault_row["voltage"],
        fault_row["current"],
        fault_row["temperature"],
        fault_row["resistance"],
        fault_row["insulation_resistance"],
        fault_row["cable_age"],
        fault_row["cable_length_km"],
        fault_row["fault_distance_km"],
    ]

    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.bar(labels, values, color=["#ef4444" if i in [1, 2, 3] else "#3b82f6" for i in range(len(labels))])
    ax.set_title(f"Faulty Cable ID {int(fault_row['cable_id'])} Feature Profile", fontsize=10)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_fault_page(dataset: pd.DataFrame):
    st.markdown("## 🛑 Faulty Cables")
    fault_df = dataset[dataset["label"] == 1].copy()
    fault_df["fault_type"] = fault_df.apply(classify_fault_type, axis=1)
    st.info(f"Dataset rows analyzed: {len(dataset)}")
    if fault_df.empty:
        st.success("No faulty cable found in this dataset.")
        return

    st.error(f"Faulty cables found: {len(fault_df)}", icon="🚨")
    st.markdown("### Faulty Cable Data")
    st.dataframe(fault_df[["cable_id", "fault_type"] + FEATURE_COLUMNS + ["label"]], use_container_width=True)

    st.markdown("### Fault Visualization Images")
    for _, row in fault_df.iterrows():
        with st.expander(f"Cable ID {int(row['cable_id'])} Fault Image", expanded=False):
            st.write(
                {
                    "cable_id": int(row["cable_id"]),
                    "voltage": float(row["voltage"]),
                    "current": float(row["current"]),
                    "temperature": float(row["temperature"]),
                    "resistance": float(row["resistance"]),
                    "insulation_resistance": float(row["insulation_resistance"]),
                    "cable_age": float(row["cable_age"]),
                    "cable_length_km": float(row["cable_length_km"]),
                    "fault_distance_km": float(row["fault_distance_km"]),
                    "fault_type": row["fault_type"],
                }
            )
            plot_fault_image(row)


def main():
    init_session_state()

    try:
        model, scaler = load_models()
    except Exception as exc:
        st.error(f"Failed to load model/scaler: {exc}")
        return

    try:
        dataset = load_dataset()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return

    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Choose page", ["Dashboard", "Safe Cables", "Faulty Cables"])
    st.sidebar.caption("IoT Data Streaming + Fault Analytics")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Alert Automation")
    st.sidebar.info("WhatsApp voice calling API is not available directly. This app sends a WhatsApp message and places a phone call using Twilio.")
    st.sidebar.caption("For WhatsApp Sandbox, From is usually +14155238886 and To must join sandbox first.")
    st.session_state["notify_enabled"] = st.sidebar.checkbox("Enable automated WhatsApp + call alerts", value=False)
    st.session_state["twilio_sid"] = st.sidebar.text_input("Twilio SID", value=os.getenv("TWILIO_ACCOUNT_SID", ""))
    st.session_state["twilio_token"] = st.sidebar.text_input("Twilio Token", value=os.getenv("TWILIO_AUTH_TOKEN", ""), type="password")
    st.session_state["from_whatsapp"] = st.sidebar.text_input("Twilio WhatsApp From (with +)", value=os.getenv("TWILIO_WHATSAPP_FROM", ""))
    st.session_state["to_whatsapp"] = st.sidebar.text_input(
        "WhatsApp Recipients (comma-separated, with +)",
        value=os.getenv("TWILIO_WHATSAPP_TO", ""),
    )
    st.session_state["use_template_mode"] = st.sidebar.checkbox("Use Twilio Template Mode (ContentSid)", value=st.session_state.get("use_template_mode", False))
    st.session_state["twilio_content_sid"] = st.sidebar.text_input("Twilio ContentSid", value=st.session_state.get("twilio_content_sid", ""))
    st.session_state["twilio_content_variables"] = st.sidebar.text_area(
        "Twilio ContentVariables JSON",
        value=st.session_state.get("twilio_content_variables", '{"1":"FAULT","2":"HIGH","3":"Cable issue"}'),
        height=90,
    )
    st.session_state["from_call"] = st.sidebar.text_input("Twilio Call From (with +)", value=os.getenv("TWILIO_CALL_FROM", ""))
    st.session_state["to_call"] = st.sidebar.text_input("Your Call To (with +)", value=os.getenv("TWILIO_CALL_TO", ""))
    if st.sidebar.button("Validate Twilio Credentials"):
        validate_twilio_credentials()
    if st.sidebar.button("Send Test WhatsApp Now"):
        send_test_whatsapp_message()
    if st.sidebar.button("Check Last Twilio Delivery Status"):
        fetch_last_message_status()
    if st.sidebar.button("Reset Seen Fault Alerts"):
        st.session_state["seen_fault_ids"] = set()
        st.session_state["seen_fault_keys"] = set()
        st.session_state["alert_log"] = []
        st.session_state["notify_status"] = "Reset completed. Alerts will send again for new faulty cables."
    if st.session_state.get("notify_status"):
        st.sidebar.write(st.session_state["notify_status"])
    if st.session_state.get("alert_log"):
        st.sidebar.markdown("### Alert Log")
        st.sidebar.write("\n".join(st.session_state["alert_log"][-10:]))

    if page == "Dashboard":
        render_dashboard(model, scaler, dataset)
    elif page == "Safe Cables":
        render_safe_page(dataset)
    else:
        render_fault_page(dataset)


if __name__ == "__main__":
    main()