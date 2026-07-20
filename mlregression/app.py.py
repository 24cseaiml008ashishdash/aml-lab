import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# --- 1. Python MLR Calculation ---
X = np.array([
    [2, 70, 12],
    [4, 80, 15],
    [6, 85, 18],
    [8, 90, 20],
    [10, 95, 22]
])
y = np.array([45, 60, 75, 90, 98])

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coefs = model.coef_
r2 = model.score(X, y)

# --- 2. Generate Graphs & Convert to Base64 ---
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

study_hours = X[:, 0]
attendance = X[:, 1]
assignment = X[:, 2]
preds = model.predict(X)

# Chart 1: Actual vs Predicted
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y, preds, color='black', label='Model Predictions')
ax.plot([40, 100], [40, 100], color='gray', linestyle='--', label='Perfect Fit (y=x)')
ax.set_xlabel('Actual Marks')
ax.set_ylabel('Predicted Marks')
ax.set_title('Actual vs Predicted Final Marks')
ax.legend()
img1 = fig_to_base64(fig)

# Chart 2: Study Hours vs Final Marks
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(study_hours, y, color='black')
ax.set_xlabel('Study Hours')
ax.set_ylabel('Final Marks')
ax.set_title('Study Hours vs Final Marks')
img2 = fig_to_base64(fig)

# Chart 3: Assignment Score vs Final Marks
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(assignment, y, color='black')
ax.set_xlabel('Assignment Score')
ax.set_ylabel('Final Marks')
ax.set_title('Assignment Score vs Final Marks')
img3 = fig_to_base64(fig)


# --- 3. Beginner-Style HTML (Minimalist CSS, Basic Tags) ---
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MLR Predictor</title>
    <!-- PyScript to run Python in browser -->
    <link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/core.css">
    <script type="module" src="https://pyscript.net/releases/2024.1.1/core.js"></script>
    <style>
        /* Minimalist default beginner styling */
        body {{
            margin: 40px;
            font-family: sans-serif;
        }}
        input {{
            margin: 5px 0;
            padding: 4px;
        }}
        button {{
            padding: 5px 10px;
            margin-top: 5px;
        }}
    </style>
</head>
<body>

    <p><b>Multiple Linear Regression Equation:</b></p>
    <p>Final Marks = {intercept:.2f} + ({coefs[0]:.2f} × Study Hours) + ({coefs[1]:.2f} × Attendance) + ({coefs[2]:.2f} × Assignment)</p>

    <ul>
        <li>Intercept: {intercept:.4f}</li>
        <li>Study Hours Coeff: {coefs[0]:.4f}</li>
        <li>Attendance Coeff: {coefs[1]:.4f}</li>
        <li>Assignment Coeff: {coefs[2]:.4f}</li>
        <li>R-squared: {r2:.4f}</li>
    </ul>

    <hr>

    <p><b>Predict Final Marks:</b></p>
    <p>
        Study Hours: <br>
        <input type="number" id="studyHours" value="5" step="0.1"><br>
        Attendance (%): <br>
        <input type="number" id="attendance" value="82" step="0.1"><br>
        Assignment Score: <br>
        <input type="number" id="assignment" value="16" step="0.1"><br>
        <button py-click="calculate_prediction">Calculate</button>
    </p>

    <p id="predictionResult"><b>Predicted Final Marks:</b> --</p>

    <hr>

    <!-- Beginner-friendly toggle for graphs -->
    <input type="checkbox" id="toggle-graphs">
    <label for="toggle-graphs"><b>Show Graphs</b></label>

    <div id="graphsDiv" style="display:none; margin-top: 15px;">
        <p><img src="data:image/png;base64,{img1}" width="400"></p>
        <p><img src="data:image/png;base64,{img2}" width="400"> <img src="data:image/png;base64,{img3}" width="400"></p>
    </div>

    <script>
        // Simple beginner JS just to show/hide graphs checkbox since CSS sibling selector can be tricky with basic tags
        document.getElementById('toggle-graphs').addEventListener('change', function() {{
            document.getElementById('graphsDiv').style.display = this.checked ? 'block' : 'none';
        }});
    </script>

    <!-- Python code running in browser via PyScript -->
    <script type="py">
from pyscript import document

def calculate_prediction(event):
    try:
        sh = float(document.querySelector("#studyHours").value)
        att = float(document.querySelector("#attendance").value)
        ass = float(document.querySelector("#assignment").value)
        
        intercept = {intercept}
        coef_study = {coefs[0]}
        coef_att = {coefs[1]}
        coef_ass = {coefs[2]}
        
        pred = intercept + (coef_study * sh) + (coef_att * att) + (coef_ass * ass)
        
        document.querySelector("#predictionResult").innerHTML = f"<b>Predicted Final Marks:</b> {{pred:.2f}}"
    except Exception as e:
        document.querySelector("#predictionResult").innerText = f"Error: {{e}}"
    </script>

</body>
</html>
"""

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Beginner-style index.html successfully generated!")
