import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

zip_path = 'data_new.zip'
csv_filename = 'data_new.csv'

# Open the zip folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract the CSV file from the zip folder
    zip_ref.extract(csv_filename, path='.')

# Read the extracted CSV file into a DataFrame
df = pd.read_csv(csv_filename)

st.title("Visualization- Final Project")
st.markdown("Reut Ben-Hamo & Anva Avraham")

accidents_per_state = df['State'].value_counts().reset_index()
accidents_per_state.columns = ['State', 'Accident Count']

# Box plot
st.subheader("Box Plot")
fig = go.Figure()
severity_levels = df['Severity'].unique()
for severity in sorted(severity_levels):
    fig.add_trace(go.Box(
        y=df[df['Severity'] == severity]['Temperature(F)'],
        name=f'Severity {severity}'
    ))
fig.update_layout(
    xaxis=dict(title='Category'),
    yaxis=dict(title='Temperature'),
    title='Severity vs Temperature'
)
st.plotly_chart(fig)

# Correlation matrix
st.subheader("Correlation Matrix")
selected_columns = ['Severity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                    'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
new_df = df[selected_columns].copy()
# Compute correlation matrix
correlation_matrix = new_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
# Create a heatmap plot of the correlation matrix
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.title('Correlation Matrix')
st.pyplot(heatmap.figure)

# Bar plot
top_10_conditions = df['Weather_Condition'].value_counts().nlargest(10).index
# Filter the DataFrame to include only the top 10 weather conditions
filtered_df = df[df['Weather_Condition'].isin(top_10_conditions)]
# Group the filtered data by weather condition and severity, and count the occurrences
grouped_data = filtered_df.groupby(['Weather_Condition', 'Severity']).size().unstack()

st.subheader("Bar Plot")
fig = px.bar(grouped_data, barmode='stack')
fig.update_layout(
    xaxis=dict(title='Weather Condition'),
    yaxis=dict(title='Count'),
    title='Top 10 Weather Condition Severity Distribution'
)
st.plotly_chart(fig)

# Map
st.subheader("USA Map - Accident Count")
fig = px.choropleth(accidents_per_state,
                    locations='State',
                    locationmode="USA-states",
                    color='Accident Count',
                    color_continuous_scale='Reds',
                    scope="usa",
                    labels={'Accident Count': 'Accident Count'}
                    )
fig.update_layout(title_text='Accident Count by State')
st.plotly_chart(fig)

st.subheader("Data")
st.write(df)
