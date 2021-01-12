import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import argparse
from utilities import dir_path
from pathlib import Path
import yaml
import os
import librosa
from utilities import LogMelExtractor
import base64
import urllib.parse
import pickle
from io import BytesIO
import soundfile as sf
from dashboard.modification import Modifications
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Temporarily hide GPU from tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Temporarily hide GPU from tf
import tensorflow as tf
import dash_table

config = yaml.safe_load(open("../config.yaml"))

# constants in css https://stackoverflow.com/a/48175165
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# colors - https://colorswall.com/palette/1856/
#354551	rgb(53, 69, 81)
#1c4c74	rgb(28, 76, 116)
#349ce4	rgb(52, 156, 228)
#6cb4e4	rgb(108, 180, 228)
#648cac	rgb(100, 140, 172)
#b2b6b0	rgb(178, 182, 176)

def create_dashboard(results, models, metadata, dataset_path, dataset):
    """
    The core function of the application, where the Dash applicaiton is created given the parameters
    :param results: Dictionary containing the results (low dimension representation) for the dataset
    :param models: Dictionary containing the models.
    :param metadata: DataFrame with the metadata for all the audio clips
    :param dataset_path: The path with the audio clips are stored
    :param dataset: the Dataset name
    :return: Dash application ready to be deployed.
    """
    def get_graphs(datatype, selected_models):
        """
        Given a list of selected models, create the scatter plots for them as a list.
        :param datatype: The selected data type to be used
        :param selected_models: The selected models
        :return: List of dash boot strap components Columnns containing the Scatter Plots.
        """
        graphs = []
        for i in range(min(len(selected_models), 4)):
            arr = results[datatype][selected_models[i]]
            arr_len = arr.shape[0]
            fig = px.scatter(x=arr[:, 0], y=arr[:, 1], color=metadata['label'], custom_data=[metadata.index[:arr_len]],
                             opacity=0.7)
            fig.update_traces(marker=dict(size=4))
            # fig.update_layout(clickmode='event+select')
            col = dbc.Col(
                [
                    html.H4(selected_models[i], className="text-center", id='graph-{}-name'.format(int(i + 1))),
                    dcc.Graph(id='graph-{}'.format(int(i + 1)), figure=fig)
                ], id='graph-{}-parent'.format(int(i + 1)))
            graphs.append(col)
        for i in range(4 - min(len(selected_models), 4)):
            j = min(len(selected_models), 4) + 1 + i
            col = dbc.Col(
                [
                    html.H4('Dim Red Technique {}'.format(j), className="text-center", id='graph-{}-name'.format(j)),
                    dcc.Graph(id='graph-{}'.format(j))
                ], id='graph-{}-parent'.format(j), style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0})
            graphs.append(col)
        return graphs

    def get_figures(datatype, selected_models, selected_index=None):
        """
        Get a list of Plotly figures, given the selected models and the selected index in these models.
       :param datatype: The selected data type to be used
        :param selected_models: The selected models
        :param selected_index: selected index on the plots, if any.
        :return:
        """
        figures = []
        for i in range(min(len(selected_models), 4)):
            arr = results[datatype][selected_models[i]]
            fig = px.scatter(x=arr[:, 0], y=arr[:, 1], color=metadata['label'], custom_data=[metadata.index],
                             opacity=0.7)
            fig.update_traces(marker=dict(size=4))
            # fig.update_layout(clickmode='event+select')
            if selected_index is not None:
                highlight_scatter = go.Scatter(
                    x=[arr[selected_index, 0]],
                    y=[arr[selected_index, 1]],
                    customdata=[metadata.index[selected_index]],
                    mode='markers',
                    showlegend=False,
                    marker=dict(size=10, line={'width': 4}, color='black', symbol='circle-open', opacity=0.8)
                )
                fig.add_trace(highlight_scatter)
            figures.append(fig)
        return figures

    def get_selected_observation(c1, c2, c3, c4, selected_models, ctx):
        """
        Quick functionality the receives the context, and given this context returns the selected index in the
        clicked graph.
        :param c1: Graph 1
        :param c2: Graph 2
        :param c3:  Graph 3
        :param c4: Graph 4
        :param selected_models:  The Selected models.
        :param ctx: Context
        :return:
        """

        if not ctx.triggered:
            graph_id = 'No clicks yet'
        else:
            graph_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # We have the graph that triggered the callback, now we set the clicked data from that
        if graph_id == 'graph-1':
            clicked = c1
        elif graph_id == 'graph-2':
            clicked = c2
        elif graph_id == 'graph-3':
            clicked = c3
        elif graph_id == 'graph-4':
            clicked = c4
        else:
            clicked = None
        # We proceed to see if it is None, so we can choose on what to do next
        if clicked is not None:
            if type(clicked['points'][0]['customdata']) is type([]):
                selected_observation = clicked['points'][0]['customdata'][0]
            else:
                selected_observation = None
        else:
            selected_observation = None
        return selected_observation

    def get_spectrogram_fig(audio_waves):
        """
        Get spectrogram and heatmap figure from the given audio waves.
        :param audio_waves: Audio waves to beb transformed into spectrograms
        :return: Tuple(numpy array containing the spectrogram, Plotly figure containing the heatmap of the spectrogram)
        """
        spectrogram_rv = get_spectrogram_frow_raw_waves(audio_waves)
        fig = get_spectrogram_fig_from_spec(spectrogram_rv)
        return spectrogram_rv, fig

    def get_spectrogram_frow_raw_waves(audio_waves):
        """
        Return numpy array containing the spectrogram for the given audio waves
        :param audio_waves: audio waves to transform into spectrogram
        :return: np array
        """
        extractor = LogMelExtractor(config[dataset]['sample_rate'],
                                    config['spectrogram']['window_size'],
                                    config['spectrogram']['hop_size'],
                                    config['spectrogram']['n_bins'],
                                    config['spectrogram']['fmin'],
                                    config['spectrogram']['fmax'])
        spectrogram = extractor.transform(audio_waves).T
        spectrogram_rv = spectrogram[::-1, :]
        return spectrogram_rv

    def get_spectrogram_fig_from_spec(spectrogram):
        """
        Creates a Plotly Heatmap figure of the given spectrogram
        :param spectrogram: np array containing the spectrogram
        :return: figure containing the spectrogram
        """
        fig_layout = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Time (Seconds)",
            yaxis_title="Frequency (Hz)",
        )
        bins, length = spectrogram.shape
        length_secs = config[dataset]['duration']
        max_freq = config['spectrogram']['fmax']
        min_freq = config['spectrogram']['fmin']
        n_xticks = 6
        n_yticks = 6
        time_values = [i * length / (n_xticks - 1) for i in range(n_xticks)]
        time_text = ["{:.2f}".format(i * length_secs / (n_xticks - 1)) for i in range(n_xticks)]
        freq_values = [i * bins / (n_yticks - 1) for i in range(n_yticks)]
        freq_text = [str(int((i * (max_freq - min_freq) / (n_yticks - 1)) + min_freq))for i in range(n_yticks)][::-1]
        fig = px.imshow(spectrogram)
        # Finish with the axes
        fig.update_xaxes(
            ticktext=time_text,
            tickvals=time_values,
        )
        fig.update_yaxes(
            ticktext=freq_text,
            tickvals=freq_values,
        )
        fig.update_layout(**fig_layout)
        return fig


    def get_base64_audio_from_path(audio_path):
        """
        Read an audio file and create a base 64 encoded representation of it
        :param audio_path: Path of the audio file to read
        :return: base 64 encoded audio.
        """
        audio_encoded = base64.b64encode(open(audio_path, 'rb').read())
        audio_source = "data:audio/wav;base64,{}".format(audio_encoded.decode())
        # base64.b64decode(audio_encoded)
        return audio_source

    def get_base64_from_numpy_array(x, sample_rate):
        """
        Get base 64 encoded version of a numpy array containing raw audio waveforms
        :param x: numpy array containing raw waveforms
        :param sample_rate: Audio sample rate
        :return: base64encoded string
        """
        # create an audio "file"
        file = BytesIO()
        # write the audio data to the file
        sf.write(file, x, sample_rate, format='wav')
        # get the file content
        content = bytes(file.getbuffer())
        audio_encoded = base64.b64encode(content)
        audio_source = "data:audio/wav;base64,{}".format(audio_encoded.decode())
        return audio_source

    # Initialize possible modifications
    modifications = Modifications()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    app.css.config.serve_locally = False
    app.css.append_css(
        {'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'})
    home_page = html.Div(children=[
            html.Div([dbc.Badge("Information", color="info", className="badge-pill mr-1"),
                      "Welcome to our audio data dimensionality reduction comparison tool. Here you can choose from "
                      "multiple techniques to see how they perform on {}.".format(config['datasets'][dataset]),html.Hr(),
                ], className='mb-3 mt-3 ml-3 mr-3 lead'),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Row('Select the input data type to be used for displaying the dimensionality reduction results:', className='lead'),
                        dbc.Row([
                            dcc.RadioItems(id='radioitems-datatype',
                                           options=[{'label': config['input_types'][inp_type],
                                                     'value': inp_type} for inp_type in results],
                                           value=list(results)[0],
                                           inputStyle={"marginRight": "5px",
                                                       "marginLeft": "10px"},
                                           className='mt-3')
                                ]),
                        html.Hr(),
                        dbc.Row(
                            'Select the dimensionality reduction techniques to be observed: ',
                            className='lead'),
                        dbc.Row(dcc.Checklist(id='checklist-models',
                                              inputStyle={"marginRight": "5px",
                                                          "marginLeft": "10px"},
                                              className='mt-3'
                                              ))
                    ]),
                    dbc.Col(
                        html.Div([
                            html.P('Add an audio clip'),
                            html.P('Pre-recorded audio clips can be added to the tool. Bear in mind that regardless '
                                   'of their length, they will be modified to match the length of the audio samples '
                                   'in the dataset. Audio format and extension must be \'.wav\'. The file name '
                                   '(without extension) will be used as the label in the graphs.'),
                            html.Div([

                                dcc.Upload(
                                    dcc.Loading([
                                        dbc.Button(
                                            html.Span(
                                                "Upload Audio File",
                                                id="tooltip-target",
                                                style={"textDecoration": "underline", "cursor": "pointer"}, ),
                                            className="ml-2 mb-5 mr-3", color="primary",
                                            outline=True),
                                            html.Div(id='div_upload_loading_helper')],
                                        type="circle",
                                        color='#2c3e50',), id='upload_audio'),
                            ], className='align-self-center')
                        ])
                    ),
                ]),
                dbc.Row(dbc.Button("Plot / Update Models", color="secondary", className="mt-3 ml-5 mr-5 btn-block",
                                   size='lg',  id='button-update'))
            ], className='mb-3 mt-3 ml-5 mr-3'),
            html.Div([
                dbc.Row([
                    dbc.Col(
                        [
                            html.H4('Dim Red Technique 1', className="text-center", id='graph-1-name'),
                            dcc.Graph(id='graph-1')
                        ], id='graph-1-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0}),
                    dbc.Col(
                        [
                            html.H4('Dim Red Technique 2', className="text-center", id='graph-2-name'),
                            dcc.Graph(id='graph-2')
                        ], id='graph-2-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0}),
                ]),
                dbc.Row([
                    dbc.Col(
                        [
                            html.H4('Dim Red Technique 3', className="text-center", id='graph-3-name'),
                            dcc.Graph(id='graph-3')
                        ], id='graph-3-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0}),
                    dbc.Col(
                        [
                            html.H4('Dim Red Technique 4', className="text-center", id='graph-4-name'),
                            dcc.Graph(id='graph-4')
                        ], id='graph-4-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0}),
                ]),

            ], id='div-plots'),
            html.Div([html.A(dbc.Button("Inverse Transform (UMAP)", className="mb-5 ml-3", color="primary", size='lg',
                                        outline=True, id='button-inverse-transform'), href='/inverse_transform'),
                      dbc.Button("Remove Uploaded & Edited Clips", className="mb-5 mr-3", color="warning", size='lg',
                                 outline=True, id='button-clear-edited')
                      ], className='d-flex justify-content-between'),
            html.Div([
                dbc.Jumbotron([
                    dbc.Row([
                        dbc.Col([dcc.Graph(id='selected_audio_spectrogram')], id='selected_audio_spectrogram_col',
                                width=6),
                        dbc.Col([
                            dbc.Row([dbc.Button('EDIT', id='button_edit_selected_audio')]),
                            dbc.Row([dbc.Button('Play', id='button_play_selected_audio')])
                        ], id='button_play_audio', width=3)
                    ]), html.A(id='anchor_edit_selected_audio'), html.P(id='p_selected_observation')
                ], style={'display': 'none'})
            ], id='div_selected_audio')
        ], id='home_page')

    edit_page = html.Div(children=[
        html.H3('Edit Selected Audio Clip'),
        dcc.Graph(id='edit_spectrogram'),

    ], id='edit_page',
        style={'display': 'block', 'lineHeight': '0', 'height': '0', 'overflow': 'hidden', 'margin': 'auto'},
        className='bg-light')

    inverse_transform_page = html.Div(children=[
        html.H3('UMAP: Inverse transform', style=dict(margin='auto', textAlign='center'), className='m-4'),
        dbc.Row([
            dbc.Col(dcc.Graph(id='inverse_transform_plot'), width=9),
            dbc.Col([
            dbc.Label('Select Input Type: ', className='font-weight-bold'),
            dcc.RadioItems(id='radioitems_datatype_inverse',
                           options=[{'label': 'Spectrograms', 'value': 'spectrograms'},
                                    {'label': 'Raw Waveform', 'value': 'raw_waveforms'}],
                           value='spectrograms',
                           inputStyle={"marginRight": "5px", "marginLeft": "10px"},
                           className='mt-3'),
            dbc.Form(
                [
                    dbc.Label('Coordinates for Inverse Transform: ', className='font-weight-bold'),
                    dbc.FormGroup(
                        [
                            dbc.Label('X-Coordinate: ', className='mr-2'),
                            dbc.Input(type='number', id='inverse_x_coord')
                        ]
                    ),
                    dbc.FormGroup(
                        [
                            dbc.Label('Y-Coordinate: ', className='mr-2'),
                            dbc.Input(type='number', id='inverse_y_coord')
                        ]
                    ),

                ],
                className='m-2',)
            ], width=3)
            ], style=dict(margin='auto', width='60%')),
        html.Center(
            dcc.Loading([dbc.Button('Transform: ', id='btn_transform', color="primary", size='lg'),
                         html.Div(id='div_inverse_transform_loading_helper')],
                        type="circle",
                        color='#2c3e50',
                        )

        ),
        html.Div([], id='inverse_transform_div'),
        html.Div([html.A(dbc.Button([html.I(className='fa fa-ban'), html.Span(' Cancel')],
                                    style=dict(margin='auto', width='100%'),
                                    className='p-2'),
                         href='/home', style=dict(margin='auto'),
                         )],
                 className='m-2 rounded d-flex justify-content-start')
    ], id='inverse_transform_page',
        style={'display': 'block', 'lineHeight': '0', 'height': '0', 'overflow': 'hidden', 'margin': 'auto'},
        )

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        dbc.NavbarSimple(brand='SingleSound', color='primary',
                         className='navbar navbar-expand-lg navbar-dark bg-primary'),
        html.Div(children=[home_page, edit_page, inverse_transform_page], id='page-content')
    ], id='content')




    # Adding a simple version of 'page navigation' that actually hides/shows elements based on the url
    # https://community.plotly.com/t/how-to-pass-values-between-pages-in-dash/33739/7
    @app.callback(
        [dash.dependencies.Output(page, 'style') for page in ['home_page', 'edit_page', 'inverse_transform_page']],
        [dash.dependencies.Input('url', 'pathname')])
    def display_page(pathname):
        """
        Dispaly a page given website path by modifying the elements' style
        :param pathname: current website path
        :return: Main section styles
        """
        return_value = [{'display': 'block', 'lineHeight': '0', 'height': '0', 'overflow': 'hidden'} for _ in range(3)]

        if pathname == '/edit':
            return_value[1] = {'height': 'auto', 'display': 'inline-block', 'width': '100%'}
            return return_value
        elif pathname == '/inverse_transform':
            return_value[2] = {'height': 'auto', 'display': 'inline-block', 'width': '100%'}
            return return_value
        else:
            return_value[0] = {'height': 'auto', 'display': 'inline-block'}
            return return_value

    @app.callback(
        dash.dependencies.Output('checklist-models', 'options'),
        [dash.dependencies.Input('radioitems-datatype', 'value')]
    )
    def update_model_checklist(data_type):
        """
        Update the checklist option given the selected data type
        :param data_type: the chosen data type in the application
        :return: content for the model checklist
        """
        options = [{'label': key, 'value': key} for key in results[data_type].keys()]
        return options

    @app.callback(
        dash.dependencies.Output('div-plots', 'children'),
        [dash.dependencies.Input('button-update', 'n_clicks'),
         dash.dependencies.Input('button-clear-edited', 'n_clicks')],
        [dash.dependencies.State('checklist-models', 'value'),
         dash.dependencies.State('radioitems-datatype', 'value')]
    )
    def plot_models(n_clicks, n_clicks_clear,selected_models, datatype):
        """
        Plot the selected models, gets triggered when clear or update buttons are clicked
        :param n_clicks: Number of clips in update buttons
        :param n_clicks_clear: Number of clips in clear button
        :param selected_models: Selected models
        :param datatype: Selected data type
        :return: List containing an html with the graphs.
        """
        ctx = dash.callback_context
        children = [

            dbc.Row([
                dbc.Col(
                    [
                        html.H4('Dim Red Technique 1', className="text-center", id='graph-1-name'),
                        dcc.Graph(id='graph-1')
                    ], id='graph-1-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0}),
                dbc.Col(
                    [
                        html.H4('Dim Red Technique 2', className="text-center", id='graph-2-name'),
                        dcc.Graph(id='graph-2')
                    ], id='graph-2-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0})
            ]),
            dbc.Row([
                dbc.Col(
                    [
                        html.H4('Dim Red Technique 3', className="text-center", id='graph-3-name'),
                        dcc.Graph(id='graph-3')
                    ], id='graph-3-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0}),
                dbc.Col(
                    [
                        html.H4('Dim Red Technique 4', className="text-center", id='graph-4-name'),
                        dcc.Graph(id='graph-4')
                    ], id='graph-4-parent', style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0}),
            ]),
        ]
        if n_clicks is None and n_clicks_clear is None:
            return children
        if not ctx.triggered:
            btn_id = 'No clicks yet'
        else:
            btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if btn_id == 'button-clear-edited':
            modifications.clear_modifications(results, metadata, dataset_path)
        if selected_models is None:
            children.extend([dbc.Alert("Make sure to select at least one model from the options in"
                                       " the upper section", color="danger"),])
            return children
        graphs = get_graphs(datatype, selected_models)
        return [
            dbc.Row([
                graphs[0],
                graphs[1],
            # ]),
            # dbc.Row([
                graphs[2],
                graphs[3],
            ]),
        ]


    @app.callback(
        [dash.dependencies.Output('graph-1', 'figure'),
         dash.dependencies.Output('graph-2', 'figure'),
         dash.dependencies.Output('graph-3', 'figure'),
         dash.dependencies.Output('graph-4', 'figure')],
        [dash.dependencies.Input('graph-1', 'clickData'),
         dash.dependencies.Input('graph-2', 'clickData'),
         dash.dependencies.Input('graph-3', 'clickData'),
         dash.dependencies.Input('graph-4', 'clickData')],
         [dash.dependencies.State('checklist-models', 'value'),
         dash.dependencies.State('radioitems-datatype', 'value')]
    )
    def update_selection(c1, c2, c3, c4, selected_models, datatype):
        """
        Update graphs given the selected observation.
        :param c1: Graph 1 clicked data
        :param c2: Graph 2 clicked data
        :param c3: Graph 3 clicked data
        :param c4: Graph 4 clicked data
        :param selected_models: Selected models
        :param datatype: Selected data type
        :return: returns a tuple with the 4 updated graphs
        """
        # Determine which figure was clicked, and determine if there are any selected observations for that figure,
        ctx = dash.callback_context
        if selected_models is None:
            return [go.Figure(data=[go.Scatter(x=[], y=[])])]*4
        selected_observation = get_selected_observation(c1, c2, c3, c4, selected_models, ctx)
        # Create the figures with the selection. If there is no select observation just return the normal plots.
        if selected_observation is None:
            figs = get_figures(datatype, selected_models)
        else:
            figs = get_figures(datatype, selected_models, selected_index=selected_observation)

        # Make sure the final returned list is length 4
        if len(figs) >= 4:
            return figs[:4]
        else:
            figs.extend([go.Figure(data=[go.Scatter(x=[], y=[])])]*(4 - len(figs)))
            return figs

    @app.callback(
        [dash.dependencies.Output('div_selected_audio', 'children')],
        [dash.dependencies.Input('graph-1', 'clickData'),
         dash.dependencies.Input('graph-2', 'clickData'),
         dash.dependencies.Input('graph-3', 'clickData'),
         dash.dependencies.Input('graph-4', 'clickData')],
        [dash.dependencies.State('checklist-models', 'value'),
         dash.dependencies.State('radioitems-datatype', 'value')]
    )
    def update_selected_audio_clip(c1, c2, c3, c4, selected_models, datatype):
        """
        Update the audio detail section given the selected observation.
        :param c1: Graph 1 clicked data
        :param c2: Graph 2 clicked data
        :param c3: Graph 3 clicked data
        :param c4: Graph 4 clicked data
        :param selected_models: Selected models
        :param datatype: Selected data type
        :return: The html element containing the detail for the selected audio clip.
        """
        # Determine which figure was clicked, and determine if there are any selected observations for that figure,
        ctx = dash.callback_context
        selected_observation = get_selected_observation(c1, c2, c3, c4, selected_models, ctx)

        if selected_observation is not None:
            if dataset == 'emotion_embeddings':
                selected_metadata = metadata.loc[selected_observation, :]
                columns = [{'name': col, 'id': col} for col in metadata.columns]
                data = [selected_metadata.to_dict()]
                children = [
                    dbc.Jumbotron([
                        dash_table.DataTable(
                            id='table',
                            columns=columns,
                            data=data
                        )
                    ], style=dict(margin='auto', width='95%'), className='p-0')
                ]
            else:
                # Read audio file
                audio_path = Path(dataset_path, metadata.filename[selected_observation])
                y, sr = librosa.load(audio_path, sr=None)
                _, fig = get_spectrogram_fig(y)

                audio_source = get_base64_audio_from_path(audio_path)
                children = [dbc.Jumbotron([
                    # html.H2('Selected Audio Clip', style={'textAlign': 'center'}, className='p-3'),
                    # html.Hr(style=dict(width='95%')),
                    dbc.Row([
                        dbc.Col([html.H4('Selected Audio Clip Spectrogram', style={'textAlign': 'center'}, className='mt-2'),
                                 dcc.Graph(id='selected_audio_spectrogram', figure=fig)],
                                id='selected_audio_spectrogram_col', width=9),
                        dbc.Col([
                            dbc.Row(html.P(html.B('File name'))),
                            dbc.Row(metadata.filename[selected_observation], className='pr-4 pl-2'),#, style={'text-align': 'center'}),
                            dbc.Row(html.P('')),
                            dbc.Row(html.P(html.B('Label'))),
                            dbc.Row(metadata.label[selected_observation], className='pr-4 pl-2'),# style={'text-align': 'center'}),
                            dbc.Row(html.P('')),
                            dbc.Row(html.Audio(src=audio_source, #src=str(audio_path),
                                               controls=True,
                                               autoPlay=False, style=dict(margin='auto', width='50%'))),
                            dbc.Row(html.P('')),
                            dbc.Row(html.A(dbc.Button([html.I(id='edit-button', className='fa fa-edit'),
                                                       html.Span(' Edit')],
                                                      id='button_edit_selected_audio', style=dict(margin='auto',
                                                                                                  width='100%'),
                                                      className='p-2'),
                                           href='/edit?i={}'.format(selected_observation), style=dict(margin='auto', width='50%'),
                                           id='anchor_edit_selected_audio')),
                            # html.P(selected_observation, id='p_selected_observation',
                            #        style={'visibility': 'hidden', 'maxHeight': 0, 'maxWidth': 0})
                        ], id='button_play_audio', width=3, style=dict(margin='auto')),
                        # html.Audio(src=str(audio_path), controls=True, autoPlay=False, type="audio/wav")
                    ], style=dict(width='95%', height='80%'))
                ], style=dict(margin='auto', width='95%'), className='p-0')
                ]

        else:
            children = [dbc.Jumbotron([
                dbc.Row([
                    dbc.Col([dcc.Graph(id='selected_audio_spectrogram')], id='selected_audio_spectrogram_col', width=9),
                    dbc.Col([
                        dbc.Row([dbc.Button('EDIT', id='button_edit_selected_audio')]),
                        dbc.Row([dbc.Button('Play', id='button_play_selected_audio')])
                    ], id='button_play_audio', width=3)
                ]), html.A(id='anchor_edit_selected_audio'),# html.P(id='p_selected_observation')
                ], style={'display': 'none'})
            ]
        return children

    @app.callback(
        [dash.dependencies.Output('edit_page', 'children')],
        [dash.dependencies.Input('url', 'search')],
        [dash.dependencies.State('edit_page', 'children')]
    )
    def edit_audio_clip(query_string, current_children):
        """
        Update the edit audio clip section, given a querystring containing the index of the audio clip to edit
        :param query_string: url query string
        :param current_children: current state of the edit page
        :return: new state of the edit page given the url parameters
        """
        if query_string is not None:
            qs_dict = dict(urllib.parse.parse_qs(query_string[1:]))
            if 'i' in qs_dict:
                selected_observation = int(qs_dict['i'][0])
                audio_path = Path(dataset_path, metadata.filename[selected_observation])
                audio_source = get_base64_audio_from_path(audio_path)
                y, sr = librosa.load(audio_path, sr=None)
                _, fig = get_spectrogram_fig(y)
                fig.update_layout(
                    autosize=False,
                    width=1000)#,
                    # height=500)
                children = [
                    html.Center([
                        html.H2('Edit Selected Audio Clip', style={'textAlign': 'center'}, className='p-3'),
                        html.Hr(style=dict(width='95%')),
                        dcc.Graph(id='edit_spectrogram', figure=fig),
                        html.Audio(src=audio_source,  # src=str(audio_path),
                                   controls=True,
                                   autoPlay=False, style=dict(margin='auto', width='40%'),
                                   id='edit_audio'),
                        # Selected modifications table.
                        html.Div(children=[], id='modification_form', className='border border-light rounded m-2'),
                        html.Div([
                            # Select modification form
                            dbc.Form(
                                [
                                    dbc.Label('Select Modification', width=4),
                                    html.Div(
                                        dcc.Dropdown(
                                            options=[dict(label=modifications.get_name_by_id(key),
                                                          value=key) for key in modifications.get_ids()],
                                            # options=[{"label": "Cut Clip", "value": 1},
                                            #          {"label": "Modify Amplitude", "value": 2},
                                            #          ],
                                            style={
                                                # 'width': '50%',
                                                'textAlign': 'left'
                                            },
                                            id='dropdown_modifications'
                                        ),
                                        className='w-50',

                                    ),
                                    dbc.Button("Add", color="primary", id='btn_add_modification'),
                                ],
                                inline=True,
                                className='m-2 rounded d-flex justify-content-between',
                                style={'width': '50%'}
                            ),
                            # Clear nofifications Button
                            dbc.Button('Clear Modifications', id='btn_clear_modifications', color="danger", outline=True,
                                       className='m-2'),

                        ], style=dict(width='100%'),
                            className='d-flex justify-content-between  border border-secondary'),
                            # Apply and Submit Modifications button
                        html.Div([
                            dbc.Button('Apply', id='btn_apply_edit', className='mr-1 mt-2 mb-4',
                                       color="primary", size='lg'),
                            dcc.Loading([dbc.Button('Submit Modified Clip', id='btn_submit_edit',
                                       className='mr-2 mt-2 mb-4',
                                       color="primary", size='lg'), html.Div(id='div_loading_helper')],
                                        type="circle",
                                        color='#2c3e50',
                                        # className='bg-primary'
                                        )
                        ], className='d-flex justify-content-end'),
                        html.P(),
                        html.Div(id='div_redirect')
                    ], style=dict(margin='auto', width='75%'), className='bg-white rounded'),
                ]
                return [children]

        # raise Exception("Query string is empty, will not update output")
        return current_children

    @app.callback(
        [dash.dependencies.Output('modification_form', 'children')],
        [dash.dependencies.Input('btn_add_modification', 'n_clicks'),
         dash.dependencies.Input('btn_clear_modifications', 'n_clicks')],
        [dash.dependencies.State('dropdown_modifications', 'value'),
         dash.dependencies.State('modification_form', 'children')]
    )
    def update_modifications(n_clicks_add, n_clicks_clear, dropdown_value, current_children):
        """
        Update the list of modification to apply to the currently edited audio clip. This function is triggered
        whenever the add modification or clear buttons are clicked.
        If the add modification button is clicked then a new element will be added to the current children,
        if the clear button is clicked then delete all current modifications.
        :param n_clicks_add: number of clicks in the add button
        :param n_clicks_clear: number of clicks in the clear button
        :param dropdown_value: Selected modification to be added
        :param current_children: Current list of modifications
        :return: New list of modifications.
        """
        ctx = dash.callback_context
        if n_clicks_add is None:# or type(current_children) != type([]):
            return [current_children]
        # Check if the function was triggered by the clear button, if so, remove all children
        if not ctx.triggered:
            pass
        else:
            if ctx.triggered[0]['prop_id'].split('.')[0] == 'btn_clear_modifications':
                return [[]]
        if dropdown_value is None:
            return [current_children]

        # Cut Clip: value = 1
        if isinstance(current_children, dict):
            current_children = [current_children]
        if dropdown_value == 1:
            # html.Div([
            row = dbc.Form(
                    [
                        dbc.Label(modifications.get_description_by_id(1)),
                        dbc.FormGroup(
                            [
                                dbc.Label('Start time: ', className='mr-2'),
                                dbc.Input(type='number', placeholder='Start (seconds)')
                            ]
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label('End time: ', className='mr-2'),
                                dbc.Input(type='number', placeholder='End (seconds)')
                            ]
                        ),

                    ],
                    inline=True,
                    className='m-2 pb-2 border-bottom border-light rounded d-flex justify-content-between',
                    id={
                        'type': 'modification_row',
                        'index': n_clicks_add
                    }
                )
            #])
            current_children.append(row)
        # Modify Amplitude: value=2
        elif dropdown_value == 2:
            row = dbc.Form(
                    [
                        dbc.Label(modifications.get_description_by_id(2)),
                        dbc.FormGroup(
                            [
                                dbc.Label('Multiplier: ', className='mr-2'),
                                dbc.Input(type='number')
                            ]
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label('Start time: ', className='mr-2'),
                                dbc.Input(type='number', placeholder='Start (seconds)')
                            ]
                        ),
                        dbc.FormGroup(
                            [
                                dbc.Label('End time: ', className='mr-2'),
                                dbc.Input(type='number', placeholder='End (seconds)')
                            ]
                        ),

                    ],
                    inline=True,
                    className='m-2 pb-2 border-bottom border-light rounded d-flex justify-content-between',
                    id={
                        'type': 'modification_row',
                        'index': n_clicks_add
                    }
                )

            current_children.append(row)
        else:
            pass
        return [current_children]

    # Apply modifications
    @app.callback(
        [dash.dependencies.Output('edit_spectrogram', 'figure'),
         dash.dependencies.Output('edit_audio', 'src')],
        [dash.dependencies.Input('btn_apply_edit', 'n_clicks')],
        [dash.dependencies.State('modification_form', 'children'),
         dash.dependencies.State('url', 'search')]
    )
    def apply_modifications(n_clicks, current_modifications, query_string):
        """
        Apply the current modifications to the audio clip being edited and return its new spectrogram and audio
        :param n_clicks: number of clicks in the apply modifications button
        :param current_modifications: Current list of modifications to be applied
        :param query_string: query string containing the id of the clip to be modified.
        :return: list [edited spectrogram figure, edited audio source]
        """
        if n_clicks is None:
            raise Exception('No clicks yet.')
        if query_string is not None:
            qs_dict = dict(urllib.parse.parse_qs(query_string[1:]))
            if 'i' in qs_dict:
                selected_observation = int(qs_dict['i'][0])
                raw_waveforms, sample_rate = modifications.apply_modifications(selected_observation,
                                                                               current_modifications, metadata,
                                                                               dataset_path)
                audio_source = get_base64_from_numpy_array(raw_waveforms, sample_rate)
                _, spectrogram_fig = get_spectrogram_fig(raw_waveforms)
                return [spectrogram_fig, audio_source]
        raise('Query string should contain the id of the clip to modify.')

    # Submit modifications
    @app.callback(
        [dash.dependencies.Output('div_redirect', 'children'),
         dash.dependencies.Output('div_loading_helper', 'children')],
        [dash.dependencies.Input('btn_submit_edit', 'n_clicks')],
        [dash.dependencies.State('modification_form', 'children'),
         dash.dependencies.State('url', 'search')]
    )
    def submit_modifications(n_clicks, current_modifications, query_string):
        """
        Submit all the modifications to the existing models and data. This requires using the
        models to create the low dimension representation for the edited audio file.
        :param n_clicks: number of clicks in the apply modifications button
        :param current_modifications: Current list of modifications to be applied
        :param query_string: query string containing the id of the clip to be modified.
        :return: A redirection to the home page of the site, after adding the new edited clip to the platform.
        """
        if n_clicks is None:
            raise Exception('No clicks yet.')
        if query_string is not None:
            qs_dict = dict(urllib.parse.parse_qs(query_string[1:]))
            if 'i' in qs_dict:
                selected_observation = int(qs_dict['i'][0])
                global metadata
                metadata2 = modifications.submit_modifications(selected_observation, current_modifications, results,
                                                               models, metadata, dataset_path, config, dataset)
                # global metadata
                metadata = metadata2

                return [dcc.Location(pathname="/home", id="loc_redirect"), []]

    # Update the plot in the inverse transform section:
    @app.callback(
        [dash.dependencies.Output('inverse_transform_plot', 'figure')],
        [dash.dependencies.Input('radioitems_datatype_inverse', 'value')]
    )
    def update_inverse_transform_plot(selected_data_type):
        """
        Update the scatter plot in the inverse transform section
        depending of the selected datatype for the inverse transform
        :param selected_data_type: the selected data type
        :return: list containing the scatter plot figure
        """
        if selected_data_type is None:
            return [None]
        arr = results[selected_data_type]['UMAP']
        arr_len = arr.shape[0]
        fig = px.scatter(x=arr[:, 0], y=arr[:, 1], color=metadata['label'], custom_data=[metadata.index[:arr_len]],
                         opacity=0.7)
        fig.update_traces(marker=dict(size=4))
        return [fig]

    # Inverse Transform
    @app.callback(
        [dash.dependencies.Output('inverse_transform_div', 'children'),
         dash.dependencies.Output('div_inverse_transform_loading_helper', 'children')],
        [dash.dependencies.Input('btn_transform', 'n_clicks')],
        [dash.dependencies.State('radioitems_datatype_inverse', 'value'),
         dash.dependencies.State('inverse_x_coord', 'value'),
         dash.dependencies.State('inverse_y_coord', 'value')]
    )
    def update_inverse_transform_body(n_clicks, selected_data_type, x_coord, y_coord):
        """
        Given some inverse transformation coordinates and a selected datatype, proceed to do the inverse
        transformaton of these coordinates.
        :param n_clicks: number of clips in the transform button
        :param selected_data_type: the select data type
        :param x_coord: the x coordinate
        :param y_coord: the y coordinate
        :return: list [the html elements containing the transformed audio detail, an empty list used as a helper
         for the ui loader]
        """
        if n_clicks is None:
            return [[], []]
        if x_coord is None or y_coord is None:
            return [[dbc.Alert("Make sure to input values for X and Y coordinates", color="danger"), ], []]
        model = models[selected_data_type]['UMAP']
        arr = results[selected_data_type]['UMAP']
        inverse_transform_coords = np.array([[x_coord, y_coord]])
        euclid_distance = np.linalg.norm(arr - inverse_transform_coords, axis=1)
        closest_index = np.argmin(euclid_distance)
        if selected_data_type == 'spectrograms':
            inverse_transform_spec = model.inverse_transform(inverse_transform_coords)
            inverse_transform_spec = inverse_transform_spec.reshape(config['spectrogram']['n_bins'], -1)
            extractor = LogMelExtractor(config[dataset]['sample_rate'],
                                        config['spectrogram']['window_size'],
                                        config['spectrogram']['hop_size'],
                                        config['spectrogram']['n_bins'],
                                        config['spectrogram']['fmin'],
                                        config['spectrogram']['fmax'])
            inverse_transform_waveform = extractor.inverse_transform(inverse_transform_spec)
            inverse_transform_spec = inverse_transform_spec[::-1, :]
            inverse_transform_spec_fig = get_spectrogram_fig_from_spec(inverse_transform_spec)

        else: # selected_data_type == 'raw_waveforms'
            inverse_transform_waveform = model.inverse_transform(inverse_transform_coords).reshape(-1)
            _, inverse_transform_spec_fig = get_spectrogram_fig(inverse_transform_waveform)

        b64encoded_inverse_transform_waveform = get_base64_from_numpy_array(inverse_transform_waveform,
                                                                            config[dataset]['sample_rate'])
        # Closest spectrogram
        closest_audio_path = Path(dataset_path, metadata.filename[closest_index])
        closest_y, sr = librosa.load(closest_audio_path, sr=None)
        _, closest_spectrogram_fig = get_spectrogram_fig(closest_y)
        b64encoded_closest_y = get_base64_from_numpy_array(closest_y, sr)

        # Create the html children
        children = [
            dbc.Row([
                dbc.Col([
                    html.Center([
                        html.H4("Inverse Transform (x={:.2f}, y={:.2f})".format(x_coord, y_coord)),
                        dcc.Graph(figure=inverse_transform_spec_fig),
                        html.Audio(src=b64encoded_inverse_transform_waveform, controls=True, autoPlay=False,
                                   style=dict(margin='auto', width='50%'))
                    ])
                ]),
                dbc.Col([
                    html.Center([
                        html.H4("Closest Audio (2d)"),
                        dcc.Graph(figure=closest_spectrogram_fig),
                        html.Audio(src=b64encoded_closest_y, controls=True, autoPlay=False,
                                   style=dict(margin='auto', width='50%'))
                    ])
                ])
            ])
        ]
        return [children, []]

        # Upload Audio
    @app.callback(
        [dash.dependencies.Output('div_upload_loading_helper', 'children')],
        [dash.dependencies.Input('upload_audio', 'contents')],
        [dash.dependencies.State('upload_audio', 'filename'),
         dash.dependencies.State('upload_audio', 'last_modified')]
    )
    def upload_audio(audio_content, filename, last_modified):
        """
        Given an audio clip, upload its contents to the platform
        :param audio_content: audio clip contents
        :param filename: audio file name
        :param last_modified: audio last modified.
        :return: empty list.
        """
        if audio_content is None:
            return [[]]
        modifications.upload_audio(audio_content, filename, results, models, metadata, dataset_path, config, dataset)
        return [[]]





    return app


def load_data(model_path, preprocessed_dataset_path, dataset):
    """
    Load all the data and models following the specified dataset
    :param model_path: Path where the models and results are stored
    :param preprocessed_dataset_path: Path to the preprocessed dataset
    :param dataset: Dataset name
    :return: Tuple (dict: containing the results from the models,
                    dict: containing the models,
                    DataFrame: containing the metadata)
    """
    if dataset in ('synthetic_dataset', 'free_spoken_digits_dataset', 'emotion_embeddings'):
        results = dict()
        models = dict()
        metadata = pd.read_csv(Path(preprocessed_dataset_path, 'metadata.csv'))
        if dataset == 'emotion_embeddings':
            metadata['label'] = metadata.family_name.str[:8]
            # metadata['label'] = metadata['emotion']
            # metadata['label'] = metadata['sentiment']
        # Make sure that the label columns is object type
        elif dataset == 'synthetic_dataset':
            cols = list(metadata.columns)
            cols[-3] = 'label'
            metadata.columns = cols
        metadata['label'] = metadata['label'].astype(str)
        model_names = config['models']
        data_types = [fn for fn in os.listdir(model_path) if not fn.endswith('.pkl') and not fn.endswith('.npy')]
        for data_type in data_types:
            results[data_type] = dict()
            models[data_type] = dict()
            results_path = Path(model_path, data_type, 'results')
            models_path = Path(model_path, data_type, 'models')
            models_filenames = [fn for fn in os.listdir(models_path) if fn.endswith('.pickle') or fn.endswith('.h5')]
            results_filenames = [fn for fn in os.listdir(results_path) if fn.endswith('.npy')]
            for models_fname in models_filenames:
                if models_fname.endswith('.h5'):
                    model = tf.keras.models.load_model(Path(models_path, models_fname), compile=False)
                    clean_name = models_fname.replace('.h5', '')
                    models[data_type][model_names[clean_name]] = model
                    pass
                else:
                    with open(Path(models_path, models_fname), 'rb') as file:
                        model = pickle.load(file)
                        clean_name = models_fname.replace('.pickle', '')
                        models[data_type][model_names[clean_name]] = model
                pass
            for results_fname in results_filenames:
                arr = np.load(Path(results_path, results_fname))
                clean_name = results_fname.replace('.npy', '')
                results[data_type][model_names[clean_name]] = arr
        return results, models, metadata



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the dashboard for Audio Dimensionality Reduction.')
    parser.add_argument('--dataset_path', type=dir_path,
                        help='Specify the absolute path (folder) where the original audio dataset is stored.')
    parser.add_argument('--model_path', type=dir_path,
                        help='Specify the absolute path (folder) where the models and results for each of the '
                             'dimensionality reduction techniques are stored.')
    parser.add_argument('--preprocessed_dataset_path', type=dir_path,
                        help='Specify the absolute path (folder) containing the preprocessed audio data.')
    parser.add_argument('--dataset', choices=['free_spoken_digits_dataset', 'synthetic_dataset', 'emotion_embeddings'],
                        default='free_spoken_digits_dataset',
                        help='Specify one of the supported datasets [free_spoken_digits_dataset, synthetic_dataset, '
                             ' emotion_embeddings]')
    args = parser.parse_args()
    model_path, preprocessed_dataset_path, dataset, dataset_path = args.model_path, \
                                                                   args.preprocessed_dataset_path, \
                                                                   args.dataset, \
                                                                   args.dataset_path
    results, models, metadata = load_data(model_path, preprocessed_dataset_path, dataset)
    app = create_dashboard(results, models, metadata, dataset_path, dataset)

    app.run_server(debug=False)