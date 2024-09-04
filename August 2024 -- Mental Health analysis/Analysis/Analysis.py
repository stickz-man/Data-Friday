
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from dash import dcc, html, Dash
import dash_bootstrap_components as dbc
import flask

# Initialize Dash app with Bootstrap and Flask
server = flask.Flask(__name__)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)

# Load your dataset (replace with the actual path to your dataset)
df = pd.read_csv('C:/Users/14432/Desktop/Data Friday/Data-Friday/August 2024 -- Mental Health analysis/Mental Health Dataset.csv')

# Keep the original dataset for visualizations
df_cleaned = df.copy()

# Drop unnecessary columns like 'Timestamp' but keep 'Growing_Stress'
df_for_model = df_cleaned.drop(columns=['Timestamp'], errors='ignore')

# Perform One-Hot Encoding for all categorical columns EXCEPT 'Growing_Stress'
categorical_cols = df_for_model.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Growing_Stress')  # Ensure 'Growing_Stress' is NOT encoded
df_for_model = pd.get_dummies(df_for_model, columns=categorical_cols, drop_first=True)

# Handle any remaining missing values in df_for_model
df_for_model = df_for_model.fillna(0)  # Replace NaNs with 0

# Split data into features (X) and target (y)
X = df_for_model.drop(columns=['Growing_Stress'])  # Features for model
y = df_for_model['Growing_Stress']  # Target

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Paths to save models
model_paths = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'Random Forest': 'random_forest_model.pkl'
}

# Initialize classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Hyperparameter tuning using GridSearchCV with reduced options
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Decision Tree': {'max_depth': [5, 10], 'min_samples_split': [5, 10]},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [10, 20]}
}

best_models = {}
accuracy_scores = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}
confusion_matrices = {}
feature_importances = {}

# Check if models are already saved
for name, path in model_paths.items():
    if os.path.exists(path):
        print(f"Loading pre-trained model for {name}...")
        best_models[name] = joblib.load(path)
    else:
        print(f"Training and saving model for {name}...")
        grid_search = GridSearchCV(models[name], param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        joblib.dump(best_models[name], path)

    # Perform cross-validation
    cv_scores = cross_val_score(best_models[name], X_train, y_train, cv=5)
    print(f"Cross-validation scores for {name}: {cv_scores}")

    # Evaluate the model
    y_pred = best_models[name].predict(X_test)
    accuracy_scores[name] = accuracy_score(y_test, y_pred)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    precision_scores[name] = precision_score(y_test, y_pred, average='weighted')
    recall_scores[name] = recall_score(y_test, y_pred, average='weighted')
    f1_scores[name] = f1_score(y_test, y_pred, average='weighted')

    # Extract feature importance for tree-based models
    if name in ['Decision Tree', 'Random Forest']:
        feature_importances[name] = best_models[name].feature_importances_

# ----------------------------
# Dash Visualizations and Layout
# ----------------------------

# Occupation Distribution Plot with context
def occupation_count_layout():
    occupation_counts = df_cleaned['Occupation'].value_counts()
    fig = px.bar(
        x=occupation_counts.index,
        y=occupation_counts.values,
        labels={'x': 'Occupation', 'y': 'Count'},
        title='Occupation Distribution',
        text=occupation_counts.values,
        color_discrete_sequence=['#FF6347'],
    )
    fig.update_layout(
        xaxis_tickangle=-45, plot_bgcolor='white', title_x=0.5,
        font=dict(family='Arial, sans-serif', size=14, color='#7f7f7f'),
        yaxis=dict(title='Count'), xaxis=dict(title='Occupation')
    )

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Occupation Distribution", className="card-title text-center"),
                html.P(
                    "This chart shows the distribution of mental health-related occupations. It helps you understand which types of occupations may be more prone to mental health challenges.",
                    className="card-text text-center"),
                dcc.Graph(figure=fig)
            ])
        ], className="shadow p-3 mb-5 bg-white rounded")
    ])

# Country Analysis Plot with context
def country_analysis_layout():
    country_counts = df_cleaned['Country'].value_counts()
    fig = px.bar(country_counts, x=country_counts.index, y=country_counts.values,
                 title='Country Analysis - Affected Population by Country',
                 labels={'x': 'Country', 'y': 'Count'},
                 text=country_counts.values)
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='white', title_x=0.5,
                      font=dict(family='Arial, sans-serif', size=14, color='#7f7f7f'),
                      yaxis=dict(title='Count'), xaxis=dict(title='Country'))

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Country Analysis", className="card-title text-center"),
                html.P(
                    "This bar chart illustrates the population distribution affected by mental health issues across different countries, allowing for global insights.",
                    className="card-text text-center"),
                dcc.Graph(figure=fig)
            ])
        ], className="shadow p-3 mb-5 bg-white rounded")
    ])

# Family History Impact Layout with context
def family_history_layout():
    family_counts = df_cleaned['family_history'].value_counts()
    fig = px.bar(family_counts, x=family_counts.index, y=family_counts.values,
                 title='Family History and Mental Health Impact')
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Family History Impact", className="card-title text-center"),
                html.P(
                    "This graph shows how having a family history of mental health issues affects an individual's likelihood of experiencing mental health challenges.",
                    className="card-text text-center"),
                dcc.Graph(figure=fig)
            ])
        ], className="shadow p-3 mb-5 bg-white rounded")
    ])

# Care Options Effectiveness Layout with context
def care_options_layout():
    care_counts = df_cleaned['care_options'].value_counts()
    fig = px.pie(values=care_counts.values, names=care_counts.index, title='Effectiveness of Care Options')
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Care Options Effectiveness", className="card-title text-center"),
                html.P(
                    "This pie chart presents the effectiveness of different mental health care options as reported by individuals.",
                    className="card-text text-center"),
                dcc.Graph(figure=fig)
            ])
        ], className="shadow p-3 mb-5 bg-white rounded")
    ])

# Growing Stress and Gender Analysis Layout with context
def growing_stress_and_gender_layout():
    gender_stress = pd.crosstab(df_cleaned['Growing_Stress'], df_cleaned['Gender'])
    fig = px.bar(gender_stress, x=gender_stress.index, y=[gender_stress[col] for col in gender_stress.columns],
                 labels={'value': 'Count', 'Growing_Stress': 'Growing Stress', 'Gender': 'Gender'},
                 title='Growing Stress by Gender', barmode='group')
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Growing Stress by Gender", className="card-title text-center"),
                html.P("This chart compares the levels of growing stress experienced by individuals based on gender.",
                       className="card-text text-center"),
                dcc.Graph(figure=fig)
            ])
        ], className="shadow p-3 mb-5 bg-white rounded")
    ])


# Confusion Matrix and Evaluation Metrics for All Models
def machine_learning_layout():
    # Show accuracy, precision, recall, and F1-score for all models
    metrics_table = pd.DataFrame({
        'Model': list(accuracy_scores.keys()),
        'Accuracy': list(accuracy_scores.values()),
        'Precision': list(precision_scores.values()),
        'Recall': list(recall_scores.values()),
        'F1-Score': list(f1_scores.values())
    })
    metrics_fig = px.bar(metrics_table, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                         title='Model Evaluation Metrics', barmode='group')

    # Confusion Matrices
    confusion_figs = []
    for model_name, cm in confusion_matrices.items():
        cm_labels = [f'Class {i}' for i in range(cm.shape[0])]
        fig_cm = ff.create_annotated_heatmap(z=cm, x=cm_labels, y=cm_labels,
                                             colorscale='Viridis', showscale=True, annotation_text=cm.astype(str))
        fig_cm.update_layout(title=f'Confusion Matrix: {model_name}', xaxis_title='Predicted', yaxis_title='Actual')
        confusion_figs.append(dcc.Graph(figure=fig_cm))

    # Feature Importance Visualization (for Decision Tree and Random Forest)
    feature_importance_figs = []
    for model_name, importances in feature_importances.items():
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                title=f'Feature Importance: {model_name}')
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        feature_importance_figs.append(dcc.Graph(figure=fig_importance))

    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Model Performance", className="card-title text-center"),
                html.P("This bar chart compares the Accuracy, Precision, Recall, and F1-Score for all models.",
                       className="card-text text-center"),
                dcc.Graph(figure=metrics_fig),
                html.H3('Confusion Matrices:', className="text-center"),
                html.Div(confusion_figs),
                html.H3('Feature Importances:', className="text-center"),
                html.Div(feature_importance_figs)
            ])
        ], className="shadow p-3 mb-5 bg-white rounded")
    ])


# Reporting and Summary Layout
def reporting_layout():
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.H4("Project Summary and Findings", className="card-title text-center"),
                html.P(
                    "This section summarizes the key findings of the mental health analysis project. We explored the impact of various factors on mental health, such as occupation, family history, and country of residence. "
                    "Our machine learning models provided insights into predicting growing stress based on these factors. Below, you can review the visualizations, model performance, and the importance of different features in the models.",
                    className="card-text text-center"),
                html.P(
                    "The models trained include Logistic Regression, Decision Tree, and Random Forest. The Decision Tree and Random Forest models showed the highest performance, with precision, recall, and F1-scores indicating a high level of accuracy.",
                    className="card-text text-center"),
                html.P(
                    "Feature importance analysis revealed that certain factors, such as occupation and country of residence, played a significant role in predicting mental health outcomes. "
                    "This information can be leveraged in developing targeted mental health interventions.",
                    className="card-text text-center"),
                html.H4("Next Steps", className="card-title text-center"),
                html.P(
                    "The next steps involve deploying these models for public use, allowing real-time predictions based on user input. This deployment will include integrating the models into a web application, providing users with immediate feedback on potential mental health risks.",
                    className="card-text text-center")
            ])
        ], className="shadow p-3 mb-5 bg-white rounded")
    ])


# Dash layout with all tabs
app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Mental Health Analysis Dashboard and Prediction -- Data Friday by Majoie Ngandi",
        brand_href="https://ngandiweb.com",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    dbc.Container([
        dcc.Tabs(id="tabs", value='tab-1', className="nav nav-tabs", children=[
            dcc.Tab(label='Occupation Distribution', value='tab-1', className="nav-item",
                    children=occupation_count_layout()),
            dcc.Tab(label='Country Analysis', value='tab-2', className="nav-item", children=country_analysis_layout()),
            dcc.Tab(label='Family History Impact', value='tab-3', className="nav-item",
                    children=family_history_layout()),
            dcc.Tab(label='Care Options Effectiveness', value='tab-4', className="nav-item",
                    children=care_options_layout()),
            dcc.Tab(label='Growing Stress and Gender Analysis', value='tab-5', className="nav-item",
                    children=growing_stress_and_gender_layout()),
            dcc.Tab(label='Machine Learning Model Performance', value='tab-6', className="nav-item",
                    children=machine_learning_layout()),
            dcc.Tab(label='Project Summary and Findings', value='tab-7', className="nav-item",
                    children=reporting_layout()),
        ]),
    ], fluid=True, className="mt-4"),
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
