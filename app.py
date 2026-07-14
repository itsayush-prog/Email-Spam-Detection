import streamlit as st
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

.main-title{
    text-align:center;
    color:#1f77b4;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}

.stTextArea textarea{
    border-radius:12px;
    border:2px solid #1f77b4;
    font-size:18px;
}

.stButton>button{
    width:100%;
    background:#1f77b4;
    color:white;
    font-size:20px;
    border-radius:10px;
    height:55px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#0d5aa7;
    color:white;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="main-title">📧 Email Spam Detection</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-title">Machine Learning Based Email Spam Detection System</div>', unsafe_allow_html=True)

st.write("")

# -------------------------------
# Input
# -------------------------------
message = st.text_area(
    "✉️ Enter Email Message",
    placeholder="""Example:

Congratulations!

You have won ₹50,000.

Click here to claim your reward.""",
    height=220
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Detect Email"):

    if message.strip() == "":
        st.warning("⚠ Please enter an email message.")
    else:

        vector = vectorizer.transform([message])

        prediction = model.predict(vector)

        st.write("")

        if prediction[0] == 1:
            st.error("🚨 SPAM EMAIL DETECTED")
            st.markdown("""
### ⚠ Recommendation

- Do not click unknown links.
- Do not share OTP or passwords.
- Delete or report this email.
""")

        else:
            st.success("✅ SAFE EMAIL")
            st.balloons()

            st.markdown("""
### ✔ Recommendation

- This email appears to be safe.
- Always verify the sender before responding.
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📊 Project Information")

st.sidebar.success("Email Spam Detection")

st.sidebar.write("""
**Dataset**

SMS Spam Collection Dataset

---

**Algorithm**

Random Forest Classifier

---

**Language**

Python

---

**Framework**

Streamlit

---

**Libraries**

• Pandas

• Scikit-learn

• Joblib

• Streamlit
""")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<hr>
<div class="footer">

Developed by <b>Ayush Gupta</b><br>

BCA Data Science Project

</div>
""", unsafe_allow_html=True)
