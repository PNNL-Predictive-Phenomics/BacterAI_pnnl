from dash import Dash, html, dcc, Output, Input, State, no_update
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
from dash.dependencies import ALL
import plotly.express as px
import pandas as pd
from sklearn.metrics import mean_squared_error
import json
import base64
import io
import os

# call bootstrap theme
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# sidebar layout
sidebar = html.Div([
    html.H2("Required User Inputs", className="text-center"),
    dbc.Card([
        dbc.CardBody([
            html.H5("Create Config JSON"),
            dcc.Input(id='experiment_path', type='text', placeholder='Experiment Path'),
            dcc.Input(id='ingredients_file', type='text', placeholder='Ingredients File'),
            dcc.Input(id='grow_threshold', type='number', placeholder='Grow Threshold'),
            dcc.Input(id='nickname', type='text', placeholder='Nickname'),
            dcc.Input(id='batch_size', type='number', placeholder='Batch Size'),
            dcc.Input(id='timeout_min', type='number', placeholder='Timeout Seconds'),
            dcc.Dropdown(
                id='model_type',
                options=[{'label': 'GPR Model', 'value': 0}, {'label': 'Neural Net Model', 'value': 1}],
                value=0,
                placeholder='Model Type',
                searchable=False  # disables typing
            ),
            dcc.Dropdown(
                id='simulation_types',
                options=[
                    {'label': 'Random', 'value': 0},
                    {'label': 'Greedy', 'value': 1},
                    {'label': 'Rollout', 'value': 2},
                    {'label': 'Rollout Prob', 'value': 3}
                ],
                value=0,
                placeholder='Simulation Types',
                searchable=False  # disables typing
            ),
            dcc.Checklist(id='beyond_frontier', options=[{'label': 'Beyond Frontier?', 'value': 'beyond_frontier'}]),
            dcc.Checklist(
                options=[{'label': 'Use Unique?', 'value': 'use_unique'}],
                id='use_unique'
            ),
            dcc.Input(id='n_rollouts', type='number', placeholder='Number of Rollouts'),
            dcc.Input(id='n_bags', type='number', placeholder='Number of Bags'),
            dcc.Input(id='transfer_model_folder', type='text', placeholder='Transfer Model Folder'),
            dcc.Input(id='transfer_data_dir', type='text', placeholder='Transfer Data Directory'),
            dcc.Input(id='redo_size', type='number', placeholder='Redo Size'),
            dcc.Input(id='redo_threshold', type='number', placeholder='Redo Threshold'),
            dcc.Checklist(
                options=[{'label': 'AAs Only?', 'value': 'aas_only'}],
                id='aas_only'
            ),
            dcc.Checklist(
                options=[{'label': 'Separate Redos?', 'value': 'separate_redos'}],
                id='separate_redos'
            ),
            html.Br(),
            html.Button("Download Config JSON", id="download_btn"),
            dcc.Download(id="download_data"),
        ])
    ], style={ "marginBottom": "1rem", "padding": "20px", "borderRadius": "12px"}),
    dbc.Card([
        dbc.CardBody([
            html.H5("Ingredients JSON"),
            html.Div(
                id="ingredients_json_display",
                style={
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace",
                    "backgroundColor": "#ffd8f600",
                    "padding": "10px",
                    "borderRadius": "5px"
                }
            ),
            html.Br(),
            html.Button("Download Ingredients JSON", id="download_ingredients_btn"),
            dcc.Download(id="download_ingredients_data"),
        ])
    ], style={"marginBottom": "1rem", "padding": "20px", "borderRadius": "12px"}),
], style={"backgroundColor": "#EEEDED", "padding": "8px", "height": "100vh", "overflowY": "auto"})

# Main content layout
main_content = html.Div([
    html.H2("Data Management", className="text-center"),
    html.Hr(),
    html.Div([
        dcc.Input(id='new_folder_name', type='text', placeholder='New folder name', style={"marginRight": "10px"}),
        html.Button("Create Folder", id='create_folder_btn', style={"marginRight": "10px"}),
        dcc.Dropdown(id='folder_selector', 
                     options=[], 
                     placeholder="Select folder", 
                     style={"marginBottom": "10px"}, 
                     searchable=False),
        dcc.Upload(
            id='upload_round_data',
            children=html.Button('Upload Round Data File(s)'),
            multiple=True,
            style={"marginRight": "10px"},
        ),
        html.Button("Process Round Data", id="process_round_data"),
    ], style={"marginBottom": "20px"}),
    dcc.Store(id='folder_files_store', data={}, storage_type='local'),
    html.Div(id='folder_files_display'),
    dcc.Store(id='folders_store', data=[], storage_type='local'),
    dcc.Store(id='selected_file', data=None),
], style={"backgroundColor": "#EEEDED", "padding": "5px", "height": "100vh", "overflowY": "auto"})


right_sidebar = html.Div([
    html.H2("Model and Prediction Analysis", className="text-center"),
    html.Div([
        html.H4("Model Performance"),
        dcc.Upload(id='upload-pred-actual', multiple=False),
        html.Div(id='model-type-display', style={'marginTop': '10px', 'fontWeight': 'bold'}),
        dcc.Graph(id='pred-actual-scatter'),
        html.Hr(),
        html.H4("Ingredient vs. Predicted Heatmap"),
        dcc.Dropdown(id='xtrain_ingredient_dropdown', 
                     placeholder="Select Ingredient for Heatmap",
                     searchable= False),
        dcc.Graph(id='xtrain-ypred-heatmap')
    ], style={"marginBottom": "2rem", "backgroundColor": "#ffffff39", "padding": "20px"}),
], style={"backgroundColor": "#EEEDED", "color": "#545454", "padding": "5px", "height": "100vh", "overflowY": "auto"})

# app layout
app.layout = html.Div([
    html.Div([
        html.Img(
            src="/assets/logo.png",
            style={
                "height": "60px",
                "display": "block",
                "marginLeft": "auto",
                "marginRight": "auto",
                "marginTop": "20px",
                "marginBottom": "10px"
            }
        ),
        #html.H1("Dashboard", style={"textAlign": "center"}),
        html.Div(
            "This dashboard allows the management of BacterAI experiments and visualization of data.",
            style={"textAlign": "center", "marginBottom": "10px"}
        ),
        html.Hr(),
    ]),
    dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=3, style={"overflowY": "auto", "height": "100vh"}),
            dbc.Col(main_content, width=5, style={"overflowY": "auto", "height": "100vh"}),
            dbc.Col(right_sidebar, width=4, style={"overflowY": "auto", "height": "100vh"}),
        ], style={"height": "100vh"})
    ], fluid=True, style={"backgroundColor": "#FBFCFC", "padding": "20px"})
])


# functions for processing uploaded files and previews
def process_uploaded_files(uploaded_files):
    categories = {"model_samples": [], "batch_dp": [], "batch_meta": [], "mapped_data": [], "random_train": [], "run_metrics": [], "ingredientvspred": []}
    for file in uploaded_files:
        filename = file['filename'].lower()
        if "batch_dp" in filename:
            categories["batch_dp"].append(file)
        elif "batch_meta" in filename:
            categories["batch_meta"].append(file)
        elif "mapped_data" in filename:
            categories["mapped_data"].append(file)
        elif "random" in filename:
            categories["random_train"].append(file)
        elif "run_metrics" in filename:
            categories["run_metrics"].append(file)
        if "predictions_gpr" in filename or "predictions_neural" in filename:
            categories["model_samples"].append(file)
        if "ypredgpr" in filename or "yprednn" in filename:
            categories["ingredientvspred"].append(file)
    return categories


def file_preview(file):
    filename = file['filename']
    preview_content = None
    preview_style = {
        "overflow": "auto",
        "backgroundColor": "#ffffff36",
        "padding": "0.5rem",
        "borderRadius": "0.25rem",
        "maxHeight": "400px", 
        "maxWidth": "1000px",   
    }
    if filename.endswith('.csv'):
        content_type, content_string = file['content'].split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            preview_content = html.Div(
                DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "350px",
                        "minWidth": "800px",
                        "width": "100%",
                    },
                    style_cell={
                        "textAlign": "left",
                        "padding": "5px",
                        "minWidth": "100px",
                        "maxWidth": "200px",
                        "whiteSpace": "normal",
                    },
                    style_header={
                        "backgroundColor": "#ffffff",
                        "fontWeight": "bold"
                    },
                    page_action="none", 
                    fixed_rows={'headers': True},
                ),
                style=preview_style
            )
        except Exception as e:
            preview_content = html.Div(f"Could not preview: {e}", style=preview_style)
    elif filename.endswith('.json'):
        content_type, content_string = file['content'].split(',')
        decoded = base64.b64decode(content_string)
        try:
            json_data = json.loads(decoded.decode('utf-8'))
            pretty_json = json.dumps(json_data, indent=2)
            preview_content = html.Pre(pretty_json, style=preview_style)
        except Exception as e:
            preview_content = html.Div(f"Could not preview: {e}", style=preview_style)
    else:
        preview_content = html.Div(
            html.Em("Binary or unsupported file preview not available."),
            style=preview_style
        )
    return preview_content

def make_list(files, category=None, title=None):
    if not files:
        return ""
    cards = []
    for file in files:
        card = dbc.Card(
            dbc.CardBody([
                html.H5(title if title else file['filename'], className="card-title"),
                html.P(f"Size: {file.get('size', 'unknown size')} bytes", className="card-text"),
                file_preview(file) if file.get('content') else None,
            ]),
            style={
                "marginBottom": "1rem",
                "maxWidth": "1000px",
                "minWidth": "900px",
                "height": "500px",
                "overflow": "hidden",
                "display": "flex",
                "flexDirection": "column"
            }
        )
        cards.append(dbc.Col(card, width="auto"))
    return dbc.Row(cards, className="g-2")

# --- Callbacks for Dropdowns, Graphs, File Management and Downloads ---


@app.callback(
    Output('pred-actual-scatter', 'figure'),
    Output('model-type-display', 'children'),
    Input('upload-pred-actual', 'contents'),
    State('upload-pred-actual', 'filename'),
    Input('folder_files_store', 'data'),
    Input('folder_selector', 'value')
)
def pred_scatter(upload_contents, upload_filename, folder_files, selected_folder):
    if folder_files and selected_folder:
        model_samples = folder_files.get(selected_folder, {}).get("model_samples", [])
        for file in model_samples:
            if file['filename'].endswith('.csv'):
                content_type, content_string = file['content'].split(',', 1)
                decoded = base64.b64decode(content_string)
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    pred_col = [col for col in df.columns if 'y_pred' in col][0]
                    true_col = [col for col in df.columns if 'y_true' in col][0]
                    predicted = df[pred_col]
                    actual = df[true_col]
                    mse = mean_squared_error(actual, predicted)
                    fig = px.scatter(
                        x=actual, y=predicted,
                        labels={'x': 'True Values', 'y': 'Predicted Values'},
                        title=f'Model Performance (MSE: {mse:.4f})'
                    )
                    fig.update_traces(marker=dict(color='#00BFAE')) 
                    fig.add_shape(
                        type='line',
                        x0=actual.min(), y0=actual.min(),
                        x1=actual.max(), y1=actual.max(),
                        line=dict(color='#007A3B', dash='dash'),
                        name='Ideal'
                    )
                    if "neural" in file['filename'].lower():
                        model_type = "Model: Neural Net"
                    elif "gpr" in file['filename'].lower():
                        model_type = "Model: GPR"
                    else:
                        model_type = "Model: Unknown (please check filename)"
                    return fig, model_type
                except Exception as e:
                    return px.scatter(title=f"Error: {e}"), ""
    return {}, ""

@app.callback(
    Output('xtrain_ingredient_dropdown', 'options'),
    Input('folder_selector', 'value'),
    Input('folder_files_store', 'data')
)
def xtrain_ingredient_options(selected_folder, folder_files):
    if not selected_folder or not folder_files:
        return []
    categories = folder_files.get(selected_folder, {})
    files = categories.get("xtrain_ypred", [])
    if not files:
        return []
    file = files[0]
    content_type, content_string = file['content'].split(',', 1)
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        options = [{'label': col, 'value': col} for col in df.columns if col != 'y_pred']
        return options
    except Exception as e:
        print("Error reading X_train_with_ypred CSV:", e)
        return []

@app.callback(
    Output('xtrain-ypred-heatmap', 'figure'),
    Input('xtrain_ingredient_dropdown', 'value'),
    Input('folder_selector', 'value'),
    Input('folder_files_store', 'data')
)
def xtrain_ypred_heatmap(selected_ingredient, selected_folder, folder_files):
    if not selected_ingredient or not selected_folder or not folder_files:
        return {}
    categories = folder_files.get(selected_folder, {})
    files = categories.get("xtrain_ypred", [])
    if not files:
        return {}
    file = files[0]
    content_type, content_string = file['content'].split(',', 1)
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if selected_ingredient in df.columns and 'y_pred' in df.columns:
            fig = px.density_heatmap(
                df, x=selected_ingredient, y='y_pred',
                title=f"Heatmap: {selected_ingredient} vs y_pred",
                labels={'x': selected_ingredient, 'y': 'y_pred'},
                color_continuous_scale="Tealgrn"  
            )
            return fig
        else:
            return px.scatter(title="Selected ingredient or y_pred not found in file.")
    except Exception as e:
        return px.scatter(title=f"Error: {e}")
    

@app.callback(
    Output('folder_files_store', 'data'),
    Input('process_round_data', 'n_clicks'),
    State('upload_round_data', 'contents'),
    State('upload_round_data', 'filename'),
    State('upload_round_data', 'last_modified'),
    State('folder_selector', 'value'),
    State('folder_files_store', 'data'),
    prevent_initial_call=True
)
def update_folder_files(n_clicks, contents, filenames, last_modified, selected_folder, folder_files):
    if folder_files is None or not isinstance(folder_files, dict):
        folder_files = {}
    if not contents or not filenames:
        return folder_files
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(contents, str):
        contents = [contents]
    folder = selected_folder or "Uncategorized"
    if folder not in folder_files:
        folder_files[folder] = []
    uploaded_files = []
    for i, fn in enumerate(filenames):
        file_dict = {'filename': fn}
        if contents and len(contents) > i:
            file_dict['content'] = contents[i]
        if last_modified and isinstance(last_modified, list) and len(last_modified) > i and contents and contents[i]:
            try:
                file_dict['size'] = len(base64.b64decode(contents[i].split(',')[1]))
            except Exception:
                file_dict['size'] = 'unknown'
        uploaded_files.append(file_dict)
    categories = process_uploaded_files(uploaded_files)
    folder = selected_folder or "Uncategorized"
    folder_files[folder] = categories
    return folder_files

# Folder creation
@app.callback(
    Output('folders_store', 'data'),
    Output('folder_selector', 'options'),
    Input('create_folder_btn', 'n_clicks'),
    State('new_folder_name', 'value'),
    State('folders_store', 'data'),
    prevent_initial_call=True
)
def create_folder(n_clicks, new_folder, folders):
    if not new_folder:
        return folders, [{'label': f, 'value': f} for f in folders]
    if new_folder not in folders:
        folders = [new_folder] + folders
    return folders, [{'label': f, 'value': f} for f in folders]

# Display files in selected folder
@app.callback(
    Output('folder_files_display', 'children'),
    Input('folder_selector', 'value'),
    Input('folder_files_store', 'data')
)
def display_folder_files(selected_folder, folder_files):
    if folder_files is None or not isinstance(folder_files, dict):
        folder_files = {}
    folder = selected_folder or "Uncategorized"
    categories = folder_files.get(folder, {})
    if not categories or not any(categories.values()):
        return html.P("No files in this folder.")
    children = []
    for cat, files in categories.items():
        if files:
            children.append(html.H4(cat.replace("_", " ").title()))
            children.append(make_list(files, category=cat))
    return children


#callback to download needed json files
@app.callback(
    Output("download_data", "data"),
    Input("download_btn", "n_clicks"),
    State('experiment_path', 'value'),
    State('ingredients_file', 'value'),
    State('grow_threshold', 'value'),
    State('nickname', 'value'),
    State('batch_size', 'value'),
    State('timeout_min', 'value'),
    State('model_type', 'value'),
    State('simulation_types', 'value'),
    State('beyond_frontier', 'value'),
    State('use_unique', 'value'),
    State('n_rollouts', 'value'),
    State('n_bags', 'value'),
    State('transfer_model_folder', 'value'),
    State('transfer_data_dir', 'value'),
    State('redo_size', 'value'),
    State('redo_threshold', 'value'),
    State('aas_only', 'value'),
    State('separate_redos', 'value'),
    prevent_initial_call=True
)
def download_file(n_clicks,
                  experiment_path, ingredients_file, grow_threshold, nickname, batch_size, timeout_min,
                  model_type, simulation_types, beyond_frontier, use_unique, n_rollouts, n_bags,
                  transfer_model_folder, transfer_data_dir, redo_size, redo_threshold, aas_only, separate_redos):
    def checklist_to_bool(val):
        return bool(val and len(val) > 0)
    config_data = {
        "experiment_path": experiment_path,
        "ingredients_file": ingredients_file,
        "grow_threshold": grow_threshold,
        "nickname": nickname,
        "batch_size": batch_size,
        "timeout_min": timeout_min,
        "model_type": int(model_type) if model_type is not None else None,
        "simulation_types": int(simulation_types) if simulation_types is not None else None,
        "beyond_frontier": beyond_frontier,
        "use_unique": checklist_to_bool(use_unique),
        "n_rollouts": n_rollouts,
        "n_bags": n_bags,
        "transfer_model_folder": transfer_model_folder,
        "transfer_data_dir": transfer_data_dir,
        "redo_size": redo_size,
        "redo_threshold": redo_threshold,
        "aas_only": checklist_to_bool(aas_only),
        "separate_redos": checklist_to_bool(separate_redos),
    }
    return dict(content=json.dumps(config_data, indent=4), filename="config.json")
@app.callback(
    Output("ingredients_json_display", "children"),
    Input("ingredients_json_display", "id")
)
def display_ingredients_json(_): 
    ingredients_path = os.path.join(os.path.dirname(__file__), "ingredients.json")
    if os.path.exists(ingredients_path):
        with open(ingredients_path, "r") as f:
            content = f.read()
        return html.Pre(
            content,
            style={
                "whiteSpace": "pre-wrap",
                "fontFamily": "monospace",
                "backgroundColor": "#ffd8f600",
                "padding": "10px",
                "borderRadius": "5px",
                "fontSize": "1rem",
                "margin": 0,
                "maxHeight": "400px",
                "overflowY": "auto"
            }
        )
    return dbc.Alert("No ingredients.json file found.", color="warning")

# Callback to download ingredients.json
@app.callback(
    Output("download_ingredients_data", "data"),
    Input("download_ingredients_btn", "n_clicks"),
    prevent_initial_call=True
)
def download_ingredients_json(n_clicks):
    ingredients_path = os.path.join(os.path.dirname(__file__), "ingredients.json")
    if os.path.exists(ingredients_path):
        with open(ingredients_path, "r") as f:
            content = f.read()
        return dict(content=content, filename="ingredients.json")
    return no_update

if __name__ == '__main__':
    app.run(debug=True)