import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Lifeboat Oracle | Titanic",
    page_icon="🛟",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# Look & feel (vintage parchment / brass boarding-pass theme)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Lato', sans-serif; }
    .block-container { padding-top: 1.5rem; max-width: 800px; }

    .masthead {
        background: #2b2118; color: #f5e6c4; border-radius: 6px;
        padding: 1.6rem 1.8rem; margin-bottom: 1.2rem; text-align: center;
        border: 3px double #d4a017;
    }
    .masthead h1 {
        margin: 0; font-family: 'Playfair Display', serif; font-size: 2.1rem;
        letter-spacing: 2px; color: #f5e6c4;
    }
    .masthead p { margin: .5rem 0 0; font-size: .95rem; color: #e7d3a5; font-style: italic; }

    .pass-heading {
        font-family: 'Playfair Display', serif; font-size: 1.15rem; color: #7a4f01;
        border-bottom: 2px dashed #d4a017; padding-bottom: .3rem; margin: .5rem 0 1rem;
    }

    .boarding-pass {
        margin-top: 1.3rem; border-radius: 8px; padding: 1.3rem 1.6rem;
        text-align: center; border: 2px dashed;
        font-family: 'Playfair Display', serif;
    }
    .boarding-pass .stamp { font-size: 1.7rem; font-weight: 700; letter-spacing: 3px; margin: 0; }
    .boarding-pass .note {
        font-family: 'Lato', sans-serif; font-size: .9rem; margin: .5rem 0 0;
    }
    .pass-saved { background: #eaf6ec; border-color: #2f855a; color: #1c4532; }
    .pass-lost  { background: #fdecea; border-color: #c0392b; color: #6b1d14; }

    div.stButton > button {
        background: #b7791f; color: #fff; border: 0; border-radius: 4px;
        padding: .65rem 1.4rem; font-weight: 700; letter-spacing: 1px; width: 100%;
    }
    div.stButton > button:hover { background: #8a5a12; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Trained model
# ---------------------------------------------------------------
@st.cache_resource
def load_titanic_model():
    with open("svc_trained_model.pkl", "rb") as fh:
        return pickle.load(fh)


titanic_model = load_titanic_model()

# ---------------------------------------------------------------
# Allowed values + label encoders (same classes as in the notebook)
# ---------------------------------------------------------------
TICKET_CLASSES = ["First", "Second", "Third"]
GENDERS = ["Male", "Female"]
SIBSP_COUNTS = ["Zero", "One", "Two", "Three"]
PORTS = ["Southampton", "Cherbourg", "Queenstown"]

enc_ticket = LabelEncoder().fit(np.ravel(TICKET_CLASSES))
enc_gender = LabelEncoder().fit(np.ravel(GENDERS))
enc_sibsp = LabelEncoder().fit(np.ravel(SIBSP_COUNTS))
enc_port = LabelEncoder().fit(np.ravel(PORTS))

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <h1>🛟 LIFEBOAT ORACLE</h1>
        <p>RMS Titanic, April 1912 &mdash; issue a boarding pass to an imaginary passenger
        and let the model decide if they reach a lifeboat.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pass, tab_info = st.tabs(["🎫 Boarding pass", "📖 Project notes"])

# ---------------------------------------------------------------
# Tab 1: passenger form + verdict
# ---------------------------------------------------------------
with tab_pass:
    st.markdown('<div class="pass-heading">Passenger particulars</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        in_ticket = st.radio("Ticket class", TICKET_CLASSES, horizontal=True)
        in_gender = st.radio("Gender", GENDERS, horizontal=True)
    with right:
        in_sibsp = st.select_slider("Siblings / spouses aboard", options=SIBSP_COUNTS)
        in_port = st.selectbox("Port of embarkation", PORTS)

    st.write("")
    go = st.button("STAMP MY BOARDING PASS")

    if go:
        raw_input = pd.DataFrame({
            "PClass": [in_ticket],
            "Gender": [in_gender],
            "Sibling": [in_sibsp],
            "Embarked": [in_port],
        })

        coded_input = raw_input.copy()
        coded_input["PClass"] = enc_ticket.transform(raw_input["PClass"])
        coded_input["Gender"] = enc_gender.transform(raw_input["Gender"])
        coded_input["Sibling"] = enc_sibsp.transform(raw_input["Sibling"])
        coded_input["Embarked"] = enc_port.transform(raw_input["Embarked"])

        verdict = titanic_model.predict(coded_input)[0]

        if verdict == 1:
            st.markdown(
                """
                <div class="boarding-pass pass-saved">
                    <p class="stamp">⛵ SURVIVED</p>
                    <p class="note">The model predicts this passenger found a place in a lifeboat.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="boarding-pass pass-lost">
                    <p class="stamp">⚓ DID NOT SURVIVE</p>
                    <p class="note">The model predicts this passenger was lost in the sinking.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Peek at the model input"):
            st.write("Entered values")
            st.dataframe(raw_input, hide_index=True)
            st.write("Label-encoded values")
            st.dataframe(coded_input, hide_index=True)

# ---------------------------------------------------------------
# Tab 2: project notes
# ---------------------------------------------------------------
with tab_info:
    st.markdown('<div class="pass-heading">About this app</div>', unsafe_allow_html=True)
    st.write(
        "A Support Vector Classifier trained on a small Titanic sample. It uses four "
        "categorical details about a passenger and predicts survival (Yes / No)."
    )
    st.table(pd.DataFrame({
        "Attribute": ["Ticket class", "Gender", "Siblings / spouses", "Port"],
        "Possible values": [
            ", ".join(TICKET_CLASSES),
            ", ".join(GENDERS),
            ", ".join(SIBSP_COUNTS),
            ", ".join(PORTS),
        ],
    }))
    st.caption("Course AIC354 - Machine Learning Fundamentals, COMSATS University Islamabad, Lahore Campus.")

st.caption("Learning project (AIC354, COMSATS Lahore).")
