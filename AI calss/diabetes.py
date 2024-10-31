import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import altair as alt
import panel as pn
from io import StringIO

pn.extension('vega')

# Load the CSV file automatically on the server-side
df = pd.read_csv('Dataset of Diabetes .csv')
df.drop(['ID', 'No_Pation'], axis=1, inplace=True)
df['CLASS'] = df['CLASS'] == 'Y'

# Function to preprocess the data
def preprocess_data(df):
    X = df[['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    df_new = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), df['CLASS']], axis=1)
    df_new['cluster'] = kmeans.labels_
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['CLASS'] = df['CLASS']
    pca_df['cluster'] = kmeans.labels_
    return df_new, pca_df

# Function to generate charts
def create_charts(df, pca_df):
    # PCA scatter plot colored by diabetes status
    chart1 = alt.Chart(pca_df).mark_circle().encode(
        x='PC1',
        y='PC2',
        color='CLASS:N',
        tooltip=['PC1', 'PC2', 'CLASS']
    ).properties(
        title='PCA Scatter Plot Colored by Diabetes Status'
    ).interactive()

    # PCA scatter plot colored by cluster
    chart2 = alt.Chart(pca_df).mark_circle().encode(
        x='PC1',
        y='PC2',
        color='cluster:N',
        tooltip=['PC1', 'PC2', 'cluster']
    ).properties(
        title='PCA Scatter Plot Colored by Cluster'
    ).interactive()

    # Histogram of age
    chart3 = alt.Chart(df).mark_bar().encode(
        x=alt.X('AGE', bin=True),
        y='count()',
        color='CLASS:N'
    ).properties(
        title='Histogram of Age'
    )

    # Box plot of HbA1c
    chart4 = alt.Chart(df).mark_boxplot().encode(
        y='HbA1c',
        color='CLASS:N'
    ).properties(
        title='Box Plot of HbA1c'
    )

    # Scatter plot of Chol vs TG
    chart5 = alt.Chart(df).mark_circle().encode(
        x='Chol',
        y='TG',
        color='CLASS:N',
        tooltip=['Chol', 'TG', 'CLASS']
    ).properties(
        title='Scatter Plot of Chol vs TG'
    ).interactive()

    return chart1, chart2, chart3, chart4, chart5

# Function to predict diabetes
def predict_diabetes(age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi):
    model = LogisticRegression()
    X = df_new.drop(columns=['CLASS', 'cluster'])
    y = df_new['CLASS']
    model.fit(X, y)
    input_data = np.array([[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])
    prediction = model.predict(input_data)
    return prediction[0]

# Preprocess the data
df_new, pca_df = preprocess_data(df)

# Create charts
chart1, chart2, chart3, chart4, chart5 = create_charts(df, pca_df)

# Panel widgets
file_input = pn.widgets.FileInput(accept='.csv')
age = pn.widgets.FloatInput(name='Age', value=30)
urea = pn.widgets.FloatInput(name='Urea', value=5)
cr = pn.widgets.FloatInput(name='Cr', value=1)
hba1c = pn.widgets.FloatInput(name='HbA1c', value=5)
chol = pn.widgets.FloatInput(name='Chol', value=180)
tg = pn.widgets.FloatInput(name='TG', value=150)
hdl = pn.widgets.FloatInput(name='HDL', value=50)
ldl = pn.widgets.FloatInput(name='LDL', value=100)
vldl = pn.widgets.FloatInput(name='VLDL', value=30)
bmi = pn.widgets.FloatInput(name='BMI', value=25)
predict_button = pn.widgets.Button(name='Predict Diabetes', button_type='primary')
output = pn.pane.Markdown('### Prediction Result')

# Function to handle file upload and processing
def process_file(event):
    file = StringIO(file_input.value.decode('utf-8'))
    global df_new, pca_df
    df = pd.read_csv(file)
    df_new, pca_df = preprocess_data(df)
    chart1, chart2, chart3, chart4, chart5 = create_charts(df, pca_df)
    charts_pane[0] = pn.Column(pn.pane.Vega(chart1), pn.pane.Vega(chart2), pn.pane.Vega(chart3), pn.pane.Vega(chart4), pn.pane.Vega(chart5))

# Function to handle prediction
def make_prediction(event):
    result = predict_diabetes(age.value, urea.value, cr.value, hba1c.value, chol.value, tg.value, hdl.value, ldl.value, vldl.value, bmi.value)
    output.object = '### Diabetic' if result else '### Not Diabetic'

file_input.param.watch(process_file, 'value')
predict_button.on_click(make_prediction)

# Layout
input_widgets = pn.Column(file_input, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi, predict_button, output)
charts_pane = pn.Column(pn.pane.Vega(chart1), pn.pane.Vega(chart2), pn.pane.Vega(chart3), pn.pane.Vega(chart4), pn.pane.Vega(chart5))

dashboard = pn.Row(input_widgets, charts_pane)
dashboard.servable()

# To run the server, execute: panel serve <this_script>.py --show
