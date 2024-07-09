# covid-19
# COVID-19 Research Dashboard

## Overview

This dashboard provides a comprehensive overview of COVID-19-related research, showcasing the latest publications, key research topics, prominent research organizations, and trends in research activity over time. The dashboard is built using Plotly Dash and is styled with the Darkly Bootstrap theme.

## Features

1. **Introduction**: An overview of the dashboard's objectives, data sources, and usage instructions.
2. **Top Publications**: Displays the most significant COVID-19 publications based on Times Cited or Altmetric Score.
3. **Topical Landscape of COVID-19 Research**: Visualizes key research topics and themes using topic modeling and a word cloud.
4. **Research Organizations**: Identifies leading research organizations and their contributions to COVID-19 and vaccine research.
5. **COVID-19 Trends Over Time**: Analyzes trends in research activity over time, including publications and grants.

## Analytical Questions Addressed

1. **Leading Research Organizations**: Identifies the top research organizations pursuing COVID-19 and vaccine research.
2. **Topical Landscape**: Visualizes the key research topics related to COVID-19.
3. **Top Publications**: Lists the top 5 most significant COVID-19 publications.
4. **Trends Over Time**: Shows how trends in COVID-19 research have changed over time.

## Installation

### Prerequisites

- Python 3.6 or higher
- Pip (Python package installer)

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/covid19-research-dashboard.git
    cd covid19-research-dashboard
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the necessary data in the required format.

5. Run the application:
    ```bash
    python app.py
    ```

6. Open a web browser and go to `http://127.0.0.1:8050/` to view the dashboard.

## File Structure

covid19-research-dashboard/
│
├── app.py # Main application file
├── db_connection.py # Database connection setup
├── requirements.txt # Required Python packages
├── assets/
│ └── wordcloud.png # Word cloud image for the topical landscape section
├── README.md # Project readme file


## Data Sources

- [Dimensions COVID-19 dataset](https://www.dimensions.ai/news/dimensions-is-facilitating-access-to-covid-19-research/)
- [GRID dataset on Google BigQuery](https://www.grid.ac/)

## Deployment

For deployment, you can package and deploy the dashboard via a cloud provider such as AWS, GCP, or Heroku. Ensure that your deployment environment has access to the required data sources and has the necessary packages installed.

## Contact

For feedback or further inquiries, please contact me at dangaichtanay@gmail.com
