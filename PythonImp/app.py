# create a dash app that uses dash bootstrap components. The app should have a figure at the 
# top and a textual input to read a formula in the variables x,y,z.

from igmap import * 
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from sympy import parse_expr

import plotly.graph_objects as go

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Create the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

presets = {
    'Sphere': {
        'formula': 'x**2 + y**2 + z**2 - 1', 
        'box': [-1,1,-1,1,-1,1]},
    'Ellipsoid': {
        'formula': 'x**2 + y**2 + (z/.5)**2 - 1',
        'box': [-1,1,-1,1,-1,1]},

    'Torus': {
        'formula': '(sqrt(x**2 + y**2) - 1)**2 + z**2 - .25**2',
        'box': [-1,1,-1,1,-1,1]
        },
}

# Create the server
server = app.server

# Create the app layout having a figure in the middle, an input to read the formula, and a button.

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Interactive Graph"),
                className="mb-2 mt-2", width=12
            )
        ),
        # add a dropdown to select a preset

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Preset"),
                        dcc.Dropdown(
                            id='preset',
                            options=[{'label': k, 'value': k} for k in presets.keys()],
                            value='Sphere',
                        ),
                    ],
                    md=4,
                ),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="graph"),
                    ],
                    md=8,
                ),
            ],
            align="center",
        ),
        dbc.Row([
            dbc.Col(
                    [
                        html.H5("Formula"),
                        dbc.Input(id="formula", type="text", placeholder="Enter a formula"),
                        dbc.Button("Submit", id="submit-button", color="primary", className="mt-2 mb-3"),
                    ],
                    md=4,
                ),
        ]),
    ],
    fluid=True,
)

# Create the callback for the button. The callback should take the formula and create a figure
# using the formula. The formula should be parsed and the variables should be replaced with
# the values of x, y, and z.

@app.callback(
    Output("graph", "figure"),
    [Input("submit-button", "n_clicks")],
    [State("formula", "value")],
)
def update_graph(n_clicks, equation):
    if n_clicks is None:
        raise PreventUpdate
    formula = parse_expr(equation)
    box = [-3,3,-3,3,-3.0,3.0]
    f, fx, fy, fz = compute_jet(formula)
    srf_vs0, srf_fs0 = iso_surface(f, box)
    srf_graph = go.Mesh3d(x=srf_vs0[:,0], y=srf_vs0[:,1], z=srf_vs0[:,2], i=srf_fs0[:,0], j=srf_fs0[:,1], k=srf_fs0[:,2], color='blue', opacity=0.25)
    

    # Create the figure
    fig = go.Figure()
    fig.add_trace(srf_graph)
    # set equal aspect ratio
    fig.update_layout(scene_aspectmode='data')
    fig.update_layout(width=800, height=800)
    return fig


# add callback for the dropdown that updates the formula
@app.callback(
    Output("formula", "value"),
    Output("submit-button", "n_clicks"),
    [Input("preset", "value")],
)
def update_formula(preset):
    return presets[preset]['formula'], 1

if __name__ == "__main__":
    app.run_server(debug=True)



