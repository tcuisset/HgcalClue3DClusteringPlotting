""" Dash components for event views : 3D and layer """

from dash import Dash, html, Input, Output, State, dcc

from event_visualizer.dash_app.plots import zAxisDropdownSettings

legendDivStyle = {'flex': '0 1 auto', 'margin':"10px"}
dropdownStyle = {'flex': '1 1 auto'}

view_3D_component = [
    # The tab div has a column flex display defined in dcc.Tabs.content_style (same for all tabs)
    html.Div(children=[
            dcc.Dropdown(id="zAxisSetting", options=zAxisDropdownSettings, value=next(iter(zAxisDropdownSettings)), style=dropdownStyle),
            dcc.Dropdown(id="projectionType", options={"perspective" : "Perspective", "orthographic" : "Orthographic"}, value="orthographic", style=dropdownStyle),
        ], 
        # The buttons div should not spread vertically, but individual buttons should spread horizontally
        style={"flex": "0 1 auto", "display" : "flex", "flexFlow":"row"}
    ),
    dcc.Loading(
        children=dcc.Graph(id="plot_3D", style={"height":"100%"}, config=dict(toImageButtonOptions=dict(
            #scale=1,
            #width=1334 * 5,
            #height=741 * 5,
        ))),
        parent_style={"flex": "1 1 auto"}, # graph should spread vertically as much as possible
    )
]
""" A 3D event view component. Graph has id 'plot_3D' """

view_layer_component = [
    # The tab div has a column flex display defined in dcc.Tabs.content_style (same for all tabs)
    html.Div(children=[
        html.Div("Layer (for layer view) :", style=legendDivStyle),
        html.Div(dcc.Slider(min=1, max=28, step=1, value=8, id="layer"), style={"flex":"10 10 auto"}), # Need a div for style=
    ], style={"flex": "0 1 auto", "display" : "flex", "flexFlow":"row"}),
    dcc.Loading(
        children=dcc.Graph(id="plot_layer", style={"height":"100%"}, config=dict(toImageButtonOptions=dict(
            height=1000,
            width=1000,
            scale=8.
        ))),
        parent_style={"flex": "1 1 auto"}, # graph should spread vertically as much as possible (note there is only one box in the flex box)
    )
]
""" A 2D event view component. Graph has id 'plot_layer' """