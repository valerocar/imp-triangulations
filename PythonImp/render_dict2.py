import dash
#import dash_table
from dash import dash_table
import pandas as pd

def render_dict(dictionary):
    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Value'])
    df.index.name = 'Key'
    df.reset_index(inplace=True)

    # Create the Dash app layout
    app = dash.Dash(__name__)
    app.layout = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        style_cell={'whiteSpace': 'normal', 'textAlign': 'left'},
        style_table={'overflowX': 'auto'},
    )

    # Run the app
    app.run_server(debug=True)

# Example dictionary
my_dict = {
    "name": "John Doe",
    "age": 30,
    "address": {
        "street": "123 Main Street",
        "city": "New York",
        "country": "USA"
    },
    "skills": {
        "programming": ["Python", "JavaScript"],
        "design": ["Photoshop", "Illustrator"]
    }
}

# Render the dictionary using Dash as a structured table
render_dict(my_dict)
