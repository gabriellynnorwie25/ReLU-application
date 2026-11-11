
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function")

st.title("ReLU Activation Function")

st.write("""
The Rectified Linear Unit (ReLU) returns 0
for negative input values and returns the input
value for positive values.
""")

st.latex(r"f(x) = \max(0, x)")

# Input slider
x_value = st.slider(
    "Choose an input value",
    -10.0, 10.0, 2.0, 0.5
)

# Calculate output
y_value = max(0, x_value)

st.metric("ReLU Output", f"{y_value:.2f}")

# Create graph
x = np.linspace(-10, 10, 400)
y = np.maximum(0, x)

fig, ax = plt.subplots()

ax.plot(x, y, label="ReLU")
ax.scatter(x_value, y_value, color="red",
           label="Selected Input")

ax.set_title("ReLU Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.subheader("Explanation")

st.write("1. Negative input produces an output of 0.")
st.write("2. Positive input produces the same output as the input.")
st.write("3. ReLU is commonly used in hidden layers of neural networks.")