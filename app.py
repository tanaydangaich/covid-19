import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from bertopic import BERTopic
import sqlite3
import os
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# Path to the database file
db_path = os.path.join(os.path.dirname(__file__), 'data', 'covid-19.db')

# Connect to the database
conn = sqlite3.connect(db_path)

# Example: Load data from the database
df_publications = pd.read_sql_query("SELECT * FROM Publications", conn)
df_clinical_trials = pd.read_sql_query("SELECT * FROM Clinical_trials", conn)
df_grants = pd.read_sql_query("SELECT * FROM Grants", conn)
df_datasets = pd.read_sql_query("SELECT * FROM Datasets", conn)

# Load the Darkly theme
external_stylesheets = [dbc.themes.DARKLY]
load_figure_template("darkly")

# Load the BERTopic model
loaded_model = BERTopic.load("path/to/my/model_dir")

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Helper function from research_orgs.py
def extract_orgs_and_metrics(df, org_col, country_col, times_cited_col, altmetric_col):
    orgs_and_metrics = []
    for org_list, country_list, times_cited, altmetric in zip(df[org_col], df[country_col], df[times_cited_col], df[altmetric_col]):
        if pd.notna(org_list) and pd.notna(country_list):
            orgs = org_list.split('; ')
            countries = country_list.split('; ')
            times_cited = int(times_cited) if pd.notna(times_cited) else 0
            altmetric = int(altmetric) if pd.notna(altmetric) else 0
            orgs_and_metrics.extend([(org, country, times_cited, altmetric) for org, country in zip(orgs, countries)])
    return orgs_and_metrics

# Significant publications from significant.py
def get_significant_publications_figure(sort_by='Timescited'):
    df_publications_sorted = df_publications.copy()
    df_publications_sorted['Timescited'] = pd.to_numeric(df_publications_sorted['Timescited'], errors='coerce').fillna(0)
    df_publications_sorted['Altmetric'] = pd.to_numeric(df_publications_sorted['Altmetric'], errors='coerce').fillna(0)

    def wrap_text(text, width):
        return '<br>'.join(text[i:i+width] for i in range(0, len(text), width))

    def update_table(sort_by='Timescited'):
        if sort_by == 'Altmetric':
            df_sorted = df_publications_sorted.sort_values(by='Altmetric', ascending=False).head(5)
        else:
            df_sorted = df_publications_sorted.sort_values(by='Timescited', ascending=False).head(5)
        df_sorted['ID'] = range(1, 6)
        df_sorted['Altmetric'] = df_sorted['Altmetric'].round(2)
        df_sorted['Title'] = df_sorted['Title'].apply(lambda x: wrap_text(x, 100))
        return df_sorted[['ID', 'Title', 'PubYear', 'Timescited', 'Altmetric']]

    sorted_data = update_table(sort_by)
    fig = go.Figure(data=[go.Table(
        columnorder=[1, 2, 3, 4, 5],
        columnwidth=[50, 400, 100, 100, 100],
        header=dict(
            values=['ID', 'Title', 'Publication Year', 'Times Cited', 'Altmetric Score'],
            fill_color='rgba(68, 1, 84, 0.6)',
            align='left',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[sorted_data[col] for col in sorted_data.columns],
            fill_color=[
                ['rgba(68, 1, 84, 0.2)', 'rgba(59, 82, 139, 0.2)', 'rgba(33, 145, 140, 0.2)', 'rgba(94, 201, 98, 0.2)', 'rgba(253, 231, 37, 0.2)']*5
            ],
            align='left',
            font=dict(size=16),
            height=60  # Increase cell height
        )
    )])
    fig.update_layout(
        title_text="Top Publications",
        title_font=dict(size=24, family='Arial Black'),
        coloraxis=dict(colorscale='Viridis', cmin=0, cmax=1),
        autosize=True,
        height=600  # Increase figure height
    )
    return fig

# Topic modeling visualization from topic_modeling_viz.py
def create_topic_viz():
    freq = loaded_model.get_topic_info()
    freq = freq.drop([1])

    def get_topic_scores(loaded_model, topic_index):
        topic_words = loaded_model.get_topic(topic_index)
        scores = [(word, score) for word, score in topic_words]
        return scores

    def get_top_n_topic_scores(freq, loaded_model, N):
        all_scores = []
        for i in range(N):
            topic_number = freq.iloc[i]["Topic"]
            topic_name = freq.iloc[i].get("Name", f"Topic {topic_number}")
            topic_representation = freq.iloc[i].get("Representation", "")
            topic_scores = get_topic_scores(loaded_model, topic_number)
            for word, score in topic_scores:
                all_scores.append({
                    "Topic Number": topic_number,
                    "Count": freq.iloc[i].get("Count", 0),
                    "Name": topic_name,
                    "Representation": topic_representation,
                    "Word": word,
                    "Score": score
                })
        df = pd.DataFrame(all_scores)
        df_filtered = df[df['Topic Number'] != -1]
        return df_filtered

    N = 9
    topic_scores_df = get_top_n_topic_scores(freq, loaded_model, N)
    scaler = MinMaxScaler()
    topic_scores_df['Normalized Score'] = scaler.fit_transform(topic_scores_df[['Score']])
    titles = [
        "Drug Discovery and Development",
        "Environmental Impact on Health",
        "Ocular Health and Diseases",
        "Immunology and Diagnostic Testing",
        "Genomics and Genetic Variability",
        "Maternal and Neonatal Health",
        "Chest Imaging and Diagnosis",
        "Sports and Athletic Performance"
    ]
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=titles,
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}], 
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.15
    )
    topic_index = 1
    for row in range(1, 3):
        for col in range(1, 5):
            if topic_index > 8:
                break
            topic_df = topic_scores_df[topic_scores_df['Topic Number'] == topic_index][:5].sort_values(by='Score', ascending=True)
            if not topic_df.empty:
                bar = go.Bar(
                    x=topic_df['Score'],
                    y=topic_df['Word'],
                    orientation='h',
                    marker=dict(color=topic_df['Normalized Score'], coloraxis="coloraxis"),
                    width=0.8
                )
                fig.add_trace(bar, row=row, col=col)
            topic_index += 1
    fig.update_layout(
        height=800,
        title_text="Topical Landscape of Covid-19 Research",
        title_font=dict(size=24, family='Arial Black'),
        coloraxis=dict(colorscale='Viridis', cmin=0, cmax=1),
        showlegend=False,
        bargap=0.2
    )
    return fig

# Research organizations visualization from research_orgs.py
def create_research_orgs_viz(org_type, sort_by):
    if org_type == 'General':
        df = df_publications
    elif org_type == 'Vaccine':
        df = df_publications[(df_publications['Title'].str.contains('vaccine', case=False, na=False)) |
                             (df_publications['Abstract'].str.contains('vaccine', case=False, na=False))]
    
    all_orgs_and_metrics = extract_orgs_and_metrics(
        df, 
        'ResearchOrganizations', 
        'CountryofResearchorganization', 
        'Timescited', 
        'Altmetric'
    )
    df_orgs_and_metrics = pd.DataFrame(all_orgs_and_metrics, columns=['ResearchOrganization', 'Country', 'TimesCited', 'Altmetric'])
    df_agg_metrics = df_orgs_and_metrics.groupby(['ResearchOrganization', 'Country']).agg(
        TotalTimesCited=('TimesCited', 'sum'), 
        TotalAltmetric=('Altmetric', 'sum')
    ).reset_index()
    
    if sort_by == 'TimesCited':
        df_agg_metrics = df_agg_metrics.sort_values(by='TotalTimesCited', ascending=False)
    elif sort_by == 'Altmetric':
        df_agg_metrics = df_agg_metrics.sort_values(by='TotalAltmetric', ascending=False)

    title = f'Top {"Vaccine " if org_type == "Vaccine" else ""}Research Organizations by {"Times Cited" if sort_by == "TimesCited" else "Altmetric Score"}'
    bar_fig = px.bar(df_agg_metrics.head(10), 
                 x='TotalTimesCited' if sort_by == 'TimesCited' else 'TotalAltmetric', 
                 y='ResearchOrganization', 
                 color='Country', 
                 orientation='h',
                 title=title,
                 labels={'TotalTimesCited': 'Total Times Cited', 'TotalAltmetric': 'Total Altmetric Score', 'ResearchOrganization': 'Research Organization'})
    bar_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    pie_fig = px.pie(df_agg_metrics.head(10), 
                     names='Country', 
                     values='TotalTimesCited' if sort_by == 'TimesCited' else 'TotalAltmetric', 
                     title=f'Country Distribution by {"Times Cited" if sort_by == "TimesCited" else "Altmetric Score"}')
    
    return bar_fig, pie_fig

# Covid-19 trends over time
def create_trends_figure(trend_type):
    # Convert 'Dateadded' and 'PublicationDate' in df_publications to datetime
    df_publications['Dateadded'] = pd.to_datetime(df_publications['Dateadded'], errors='coerce')
    df_publications['PublicationDate'] = pd.to_datetime(df_publications['PublicationDate'], errors='coerce')

    # Extract year and month for aggregation
    df_publications['YearMonth'] = df_publications['PublicationDate'].dt.to_period('M')

    # Count the number of publications per month
    publications_per_month = df_publications.groupby('YearMonth').size().reset_index(name='PublicationCount')

    # Convert 'Dateadded', 'Startdate', and 'Enddate' in df_grants to datetime
    df_grants['Dateadded'] = pd.to_datetime(df_grants['Dateadded'], errors='coerce')
    df_grants['Startdate'] = pd.to_datetime(df_grants['Startdate'], errors='coerce')
    df_grants['Enddate'] = pd.to_datetime(df_grants['Enddate'], errors='coerce')

    # Extract year and month for aggregation
    df_grants['YearMonth'] = df_grants['Startdate'].dt.to_period('M')

    # Count the number of grants per month
    grants_per_month = df_grants.groupby('YearMonth').size().reset_index(name='GrantCount')

    # Convert 'YearMonth' to string for Plotly
    publications_per_month['YearMonth'] = publications_per_month['YearMonth'].astype(str)
    grants_per_month['YearMonth'] = grants_per_month['YearMonth'].astype(str)

    # Create an overlapping line chart using Plotly with secondary y-axis
    fig = go.Figure()

    if trend_type == 'Both' or trend_type == 'Publications':
        fig.add_trace(go.Scatter(x=publications_per_month['YearMonth'], y=publications_per_month['PublicationCount'],
                                 mode='lines+markers', name='Publications', yaxis='y1',
                                 hovertemplate='Date: %{x}<br>Publications: %{y}'))

    if trend_type == 'Both' or trend_type == 'Grants':
        fig.add_trace(go.Scatter(x=grants_per_month['YearMonth'], y=grants_per_month['GrantCount'],
                                 mode='lines+markers', name='Grants', yaxis='y2',
                                 hovertemplate='Date: %{x}<br>Grants: %{y}'))

    # Customize the layout
    fig.update_layout(
        title='Trends in COVID-19 Research Publications and Grants Over Time',
        xaxis=dict(
            title='Slider for time', 
            title_font=dict(size=18),  # Increase x-axis title font size
            tickfont=dict(size=14),    # Increase x-axis tick font size
            tickangle=0, 
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(
            title='Number of Publications', 
            title_font=dict(size=18),  # Increase y-axis title font size
            tickfont=dict(size=14),    # Increase y-axis tick font size
            side='left'
        ),
        yaxis2=dict(
            title='Number of Grants', 
            title_font=dict(size=18),  # Increase y-axis2 title font size
            tickfont=dict(size=14),    # Increase y-axis2 tick font size
            overlaying='y', 
            side='right'
        ),
        legend=dict(
            title='Type',
            xanchor="left",
            x=0.05),
        hovermode='x unified',
    )

    return fig

# Layout of the Dash app
app.layout = dbc.Container([
    html.H1("COVID-19 Research Dashboard", style={'text-align': 'center'}),
    dcc.Tabs(id='tabs', value='intro', children=[
        dcc.Tab(label='Introduction', value='intro', style={'backgroundColor': '#343a40', 'color': 'white'}, selected_style={'backgroundColor': '#007bff', 'color': 'white'}),
        dcc.Tab(label='Top Publications', value='tab-1', style={'backgroundColor': '#343a40', 'color': 'white'}, selected_style={'backgroundColor': '#007bff', 'color': 'white'}),
        dcc.Tab(label='Topical Landscape of Covid-19 Research', value='tab-2', style={'backgroundColor': '#343a40', 'color': 'white'}, selected_style={'backgroundColor': '#007bff', 'color': 'white'}),
        dcc.Tab(label='Research Organizations', value='tab-3', style={'backgroundColor': '#343a40', 'color': 'white'}, selected_style={'backgroundColor': '#007bff', 'color': 'white'}),
        dcc.Tab(label='Covid-19 Trends Over Time', value='tab-4', style={'backgroundColor': '#343a40', 'color': 'white'}, selected_style={'backgroundColor': '#007bff', 'color': 'white'}),
    ]),
    html.Div(id='tabs-content')
], fluid=True)

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'intro':
        return html.Div([
            html.H1("Welcome to the COVID-19 Research Dashboard", style={'text-align': 'center'}),
            html.H2("Objective"),
            html.P("This dashboard aims to provide a comprehensive overview of COVID-19 related research, showcasing the latest publications, key research topics, prominent research organizations, and trends in research activity over time."),
            html.H2("Background on COVID-19"),
            html.P("COVID-19, caused by the SARS-CoV-2 virus, emerged in late 2019 and has since led to a global pandemic. The virus has had significant health, economic, and social impacts worldwide."),
            html.H2("Data Sources"),
            html.P("The data presented in this dashboard is sourced from various scientific databases and repositories, including publications, grants, and clinical trials related to COVID-19 research."),
            html.H2("How to Use the Dashboard"),
            html.Ul([
                html.Li("Top Publications: View the most cited and impactful publications on COVID-19."),
                html.Li("Topical Landscape of Covid-19 Research: Explore the key research topics and themes."),
                html.Li("Research Organizations: Identify leading research organizations and their contributions."),
                html.Li("Covid-19 Trends Over Time: Analyze trends in research activity over time.")
            ])
        ])
    elif tab == 'tab-1':
        return html.Div([
            html.H2("Top Publications"),
            dcc.Graph(figure=get_significant_publications_figure())
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H2("Topical Landscape of Covid-19 Research"),
            dcc.Graph(figure=create_topic_viz())
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H2("Research Organizations"),
            dcc.Dropdown(
                id='org-type-dropdown',
                options=[
                    {'label': 'General Research Organizations', 'value': 'General'},
                    {'label': 'Vaccine Research Organizations', 'value': 'Vaccine'}
                ],
                value='General',
                style={'color': 'black'}  # Set text color to black
            ),
            dcc.Dropdown(
                id='sort-by-dropdown',
                options=[
                    {'label': 'Times Cited', 'value': 'TimesCited'},
                    {'label': 'Altmetric Score', 'value': 'Altmetric'}
                ],
                value='TimesCited',
                style={'color': 'black'}  # Set text color to black
            ),
            html.Div([
                dcc.Graph(id='research-orgs-bar-graph', style={'display': 'inline-block', 'width': '65%'}),
                dcc.Graph(id='research-orgs-pie-graph', style={'display': 'inline-block', 'width': '33%'})
            ])
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H2("Covid-19 Trends Over Time"),
            dcc.Dropdown(
                id='trend-type-dropdown',
                options=[
                    {'label': 'Show Both', 'value': 'Both'},
                    {'label': 'Highlight Publications', 'value': 'Publications'},
                    {'label': 'Highlight Grants', 'value': 'Grants'}
                ],
                value='Both',
                style={'color': 'black'}  # Set text color to black
            ),
            dcc.Graph(id='covid-trends-graph')
        ])

@app.callback(
    [Output('research-orgs-bar-graph', 'figure'), Output('research-orgs-pie-graph', 'figure')],
    [Input('org-type-dropdown', 'value'), Input('sort-by-dropdown', 'value')]
)
def update_research_orgs_graphs(org_type, sort_by):
    bar_fig, pie_fig = create_research_orgs_viz(org_type, sort_by)
    return bar_fig, pie_fig

@app.callback(
    Output('covid-trends-graph', 'figure'),
    Input('trend-type-dropdown', 'value')
)
def update_trends_graph(trend_type):
    return create_trends_figure(trend_type)

server = app.server
if __name__ == '__main__':
    app.run_server(debug=True)

