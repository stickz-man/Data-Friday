import dash
import pandas as pd
from dash import dcc,html
from dash.dependencies import Input, Output
import plotly.express as px
from flask import Flask

# initialize a flask server
server = Flask(__name__)

# loading dataset from location to pandas dataframe
df = pd.read_csv('C:/Users/14432/Desktop/Data science/August 2024 -- Mental Health analysis/Mental Health Dataset.csv')

# Initialize Dash app with Flask server
app = dash.Dash(__name__, server=server)

# Data Wrangling: Preprocessing data (cleaning, handling missing values, etc.)
df_cleaned = df.dropna()


app.layout = html.Div ([
    html.H1('Mental Health Analysis'), # Title for the whole analysis page
    #Dash tabs created to navigate through different pages
    dcc.Tabs(id='tabs', value='tab', children=[
        dcc.Tab(label='Country Analysis', value='tab',),
        dcc.Tab(label='Student vs Corporate', value='tab-2'),
        dcc.Tab(label='Family History Impact', value='tab-3'),
        dcc.Tab(label='Care Options Effectiveness', value='tab-4'),
        dcc.Tab(label='Growing Stress and Gender Analysis', value='tab-5'),
    ]),
    html.Div(id='tabs-content')

])


#App callback to switch between tabs
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab':
        return country_analysis_layout()
    elif tab == 'tab-2':
        return student_vs_corporate_layout()
    elif tab == 'tab-3':
        return family_history_layout()
    elif tab == 'tab-4':
        return care_options_layout()
    elif tab == 'tab-5':
        return growing_stress_layout()




# ==================== Dashboard Layouts ====================


def country_analysis_layout():
    # Visualization: Bar chart showing percentage of affected population by country
    country_counts = df_cleaned['Country'].value_counts(normalize=True) * 100
    fig = px.bar(country_counts, x=country_counts.index, y=country_counts.values,
                 title='Percentage of Affected Population by Country')

    return html.Div([
        html.H2('Country Analysis'),
        dcc.Graph(figure=fig)
    ])


def student_vs_corporate_layout():
    # Count the number of occurrences for each Occupation in the dataset
    occupation_counts = df_cleaned['Occupation'].value_counts()

    # Create a bar chart using Plotly Express
    fig = px.bar(
        x=occupation_counts.index,  # Occupations on the x-axis
        y=occupation_counts.values,  # Counts on the y-axis
        labels={'x': 'Occupation', 'y': 'Count'},
        title='Occupation Distribution',
        text=occupation_counts.values  # Display count values on top of the bars
    )

    # Update layout to match the desired style (rotating the x-axis labels)
    fig.update_layout(
        xaxis_tickangle=-45,  # Rotate the x-axis labels by 45 degrees for better readability
        plot_bgcolor='white',  # Set the background color to white
        title_x=0.5,  # Center the title
        yaxis=dict(title='Count'),  # Set the title of y-axis
        xaxis=dict(title='Occupation')  # Set the title of x-axis
    )

    # Return the bar chart in a Dash layout
    return html.Div([
        html.H2('Occupation Distribution'),
        dcc.Graph(figure=fig)
    ])


def family_history_layout():
    # Visualization: Family history impact on mental health
    family_counts = df_cleaned['family_history'].value_counts()
    fig = px.bar(family_counts, x=family_counts.index, y=family_counts.values,
                 title='Family History and Mental Health Impact')

    return html.Div([
        html.H2('Family History Impact'),
        dcc.Graph(figure=fig)
    ])


def care_options_layout():
    # Visualization: Effectiveness of care options (assuming the dataset has "Care_Options" and success rate)
    care_counts = df_cleaned['care_options'].value_counts()
    fig = px.pie(values=care_counts.values, names=care_counts.index, title='Effectiveness of Care Options')

    return html.Div([
        html.H2('Care Options Effectiveness'),
        dcc.Graph(figure=fig)
    ])


def growing_stress_layout():
    # Create a crosstab for Growing Stress by Gender
    crosstab_df = pd.crosstab(df_cleaned['Growing_Stress'], df_cleaned['Gender'])

    # Plotly Express bar chart for crosstab result (grouped bar chart)
    fig = px.bar(
        crosstab_df,
        x=crosstab_df.index,  # The Growing Stress categories
        y=crosstab_df.columns,  # The Gender categories
        barmode='group',  # Group bars by gender
        labels={'value': 'Count', 'Growing_Stress': 'Growing Stress', 'Gender': 'Gender'},
        title='Growing Stress by Gender',
        color_discrete_sequence=['#ff69b4', '#00ced1']  # 'm' (magenta) and 'c' (cyan) color equivalent
    )

    # Return the updated bar chart
    return html.Div([
        html.H2('Growing Stress by Gender'),
        dcc.Graph(figure=fig)
    ])


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
