"""
app.py
Dash application: interactively damage clean speech with noise,
then enhance it using the ai-coustics SDK.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from audio import (
    TARGET_SR, NOISE_TYPES,
    audio_to_base64_wav, compute_spectrogram,
    get_noisy_audio, load_clean_speech,
)
from enhance import (
    ALL_MODELS, available_models,
    DEFAULT_BYPASS, DEFAULT_VAD_HOLD,
    DEFAULT_VAD_MIN_SPEECH, DEFAULT_VAD_SENSITIVITY,
    enhance, preload_default_model, start_model_probe,
)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app    = Dash(__name__, title="ai-coustics Enhancement Demo")
server = app.server

preload_default_model()
start_model_probe()

_clean, _sr    = load_clean_speech()
_f0, _t0, _S0  = compute_spectrogram(_clean, _sr)
_orig_b64      = audio_to_base64_wav(_clean, _sr)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

COLORSCALE           = "Viridis"
SPEC_ZMIN, SPEC_ZMAX = -100, 20


def make_spectrogram_figure(
        f: np.ndarray, t: np.ndarray, S_db: np.ndarray,
        title: str,
        vad_curve: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=t, y=f, z=S_db,
        colorscale=COLORSCALE, zmin=SPEC_ZMIN, zmax=SPEC_ZMAX,
        colorbar=dict(title="dB", thickness=12, len=0.8),
        hoverinfo="skip",
    ))
    if vad_curve is not None and len(vad_curve) > 0:
        vad_t = np.linspace(t[0], t[-1], len(vad_curve))
        vad_y = vad_curve * float(f[-1]) * 0.92
        fig.add_trace(go.Scatter(
            x=vad_t, y=vad_y, mode="lines",
            line=dict(color="white", width=1.5, dash="dot", shape="hv"),
            name="VAD", hoverinfo="skip",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Time (s)", showgrid=False),
        yaxis=dict(title="Frequency (Hz)", showgrid=False),
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"), height=260,
        legend=dict(font=dict(size=10), x=0.01, y=0.97),
    )
    return fig


def empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="Adjust settings and press Apply",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color="#718096", size=13),
    )
    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", height=260,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

COL  = {
    "flex": "1", "padding": "16px", "display": "flex",
    "flexDirection": "column", "gap": "8px",
    "background": "#16213e", "borderRadius": "10px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.4)",
}
CARD = {"background": "#16213e", "borderRadius": "10px",
        "padding": "20px", "boxShadow": "0 2px 12px rgba(0,0,0,0.4)"}
LBL  = {"color": "#a0aec0", "fontSize": "12px", "marginBottom": "2px", "fontWeight": 600}
HINT = {"color": "#4a5568", "fontSize": "11px", "margin": "0 0 4px 0"}
DD   = {"background": "#2d3748", "color": "#e2e8f0", "border": "none"}
BTN  = {
    "background": "linear-gradient(135deg,#667eea,#764ba2)",
    "color": "white", "border": "none", "borderRadius": "8px",
    "padding": "10px 32px", "fontSize": "14px", "fontWeight": "600",
    "cursor": "pointer", "letterSpacing": "0.5px",
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"background": "#0f0f23", "minHeight": "100vh", "padding": "24px",
           "fontFamily": "Inter, system-ui, sans-serif"},
    children=[

        # Header
        html.Div(style={"textAlign": "center", "marginBottom": "28px"}, children=[
            html.H1("ai-coustics Enhancement Demo",
                    style={"color": "#e2e8f0", "fontSize": "24px", "margin": 0, "fontWeight": 600}),
            html.P("Damage clean speech with noise · Restore it with AI",
                   style={"color": "#718096", "margin": "6px 0 0 0", "fontSize": "13px"}),
        ]),

        # Controls
        html.Div(style={**CARD, "marginBottom": "20px"}, children=[

            html.Div(style={"display": "flex", "gap": "32px", "flexWrap": "wrap",
                            "marginBottom": "20px"}, children=[

                # Noise Level dBFS
                html.Div(style={"flex": "2", "minWidth": "200px"}, children=[
                    html.Label("Noise Level (dBFS)", style=LBL),
                    dcc.Slider(
                        id="noise-slider",
                        min=-80, max=0, step=1, value=-40,
                        marks={-80: "-80", -60: "-60", -40: "-40", -20: "-20", 0: "0"},
                        tooltip={"always_visible": True, "placement": "bottom"},
                    ),
                ]),

                # Noise Type
                html.Div(style={"flex": "2", "minWidth": "200px"}, children=[
                    html.Label("Noise Type (select one or more)", style=LBL),
                    dcc.Dropdown(id="noise-type-dropdown",
                                 options=[{"label": n, "value": n} for n in NOISE_TYPES],
                                 value=["White"], multi=True, clearable=False, style=DD),
                ]),

                # Enhancement Level
                html.Div(style={"flex": "2", "minWidth": "200px"}, children=[
                    html.Label("Enhancement Level", style=LBL),
                    dcc.Slider(id="enhancement-slider", min=0.0, max=1.0, step=0.05, value=0.8,
                               marks={0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1.0"},
                               tooltip={"always_visible": True, "placement": "bottom"}),
                ]),

                # Model
                html.Div(style={"flex": "1", "minWidth": "180px"}, children=[
                    html.Label("Model", style=LBL),
                    dcc.Dropdown(id="model-dropdown",
                                 options=[{"label": m, "value": m} for m in available_models],
                                 value=available_models[0],
                                 clearable=False, style=DD),
                ]),
            ]),

            # Advanced Parameters
            html.Details(style={"marginBottom": "20px"}, children=[
                html.Summary("⚙ Advanced Parameters",
                             style={"color": "#a0aec0", "fontSize": "13px",
                                    "cursor": "pointer", "userSelect": "none",
                                    "marginBottom": "16px"}),
                html.Div(style={"display": "flex", "gap": "32px", "flexWrap": "wrap",
                                "paddingTop": "12px", "borderTop": "1px solid #2d3748"}, children=[

                    html.Div(style={"flex": "1", "minWidth": "180px"}, children=[
                        html.Label("Bypass (dry/wet mix)", style={**LBL, "color": "#718096"}),
                        html.P("0 = full enhancement · 1 = dry signal", style=HINT),
                        dcc.Slider(id="bypass-slider", min=0.0, max=1.0, step=0.05,
                                   value=DEFAULT_BYPASS,
                                   marks={0: "0.0", 0.5: "0.5", 1: "1.0"},
                                   tooltip={"always_visible": True, "placement": "bottom"}),
                    ]),

                    html.Div(style={"flex": "1", "minWidth": "180px"}, children=[
                        html.Label("VAD Sensitivity", style={**LBL, "color": "#718096"}),
                        html.P("Higher = more sensitive to quiet speech", style=HINT),
                        dcc.Slider(id="vad-sensitivity-slider", min=1.0, max=10.0, step=0.5,
                                   value=DEFAULT_VAD_SENSITIVITY,
                                   marks={1: "1", 5: "5", 10: "10"},
                                   tooltip={"always_visible": True, "placement": "bottom"}),
                    ]),

                    html.Div(style={"flex": "1", "minWidth": "180px"}, children=[
                        html.Label("VAD Speech Hold (s)", style={**LBL, "color": "#718096"}),
                        html.P("How long VAD stays active after speech ends", style=HINT),
                        dcc.Slider(id="vad-hold-slider", min=0.0, max=1.0, step=0.05,
                                   value=DEFAULT_VAD_HOLD,
                                   marks={0: "0", 0.5: "0.5", 1: "1"},
                                   tooltip={"always_visible": True, "placement": "bottom"}),
                    ]),

                    html.Div(style={"flex": "1", "minWidth": "180px"}, children=[
                        html.Label("VAD Min Speech Duration (s)", style={**LBL, "color": "#718096"}),
                        html.P("Minimum length to trigger VAD detection", style=HINT),
                        dcc.Slider(id="vad-min-slider", min=0.0, max=0.5, step=0.01,
                                   value=DEFAULT_VAD_MIN_SPEECH,
                                   marks={0: "0", 0.25: "0.25", 0.5: "0.5"},
                                   tooltip={"always_visible": True, "placement": "bottom"}),
                    ]),
                ]),
            ]),

            # Apply row
            html.Div(style={"display": "flex", "alignItems": "center",
                            "justifyContent": "flex-end", "gap": "16px"}, children=[
                html.Span(id="status-text",
                          style={"color": "#718096", "fontSize": "12px"},
                          children="Adjust settings and press Apply."),
                html.Button("▶  Apply", id="apply-btn", n_clicks=0, style=BTN),
            ]),
        ]),

        # Three-column comparison
        html.Div(style={"display": "flex", "gap": "12px", "alignItems": "stretch"}, children=[

            html.Div(style=COL, children=[
                html.H3("🎙 Original", style={"color": "#68d391", "margin": 0, "fontSize": "15px"}),
                dcc.Graph(id="orig-spec",
                          figure=make_spectrogram_figure(_f0, _t0, _S0, "Clean Speech"),
                          config={"displayModeBar": False}),
                html.Audio(id="orig-audio", src=_orig_b64, controls=True,
                           style={"width": "100%", "marginTop": "8px"}),
            ]),

            html.Div(style=COL, children=[
                html.H3("🌧 Noisy", style={"color": "#fc8181", "margin": 0, "fontSize": "15px"}),
                dcc.Loading(type="dot", color="#fc8181", children=
                dcc.Graph(id="noisy-spec", figure=empty_figure(),
                          config={"displayModeBar": False})),
                html.Audio(id="noisy-audio", src="", controls=True,
                           style={"width": "100%", "marginTop": "8px"}),
            ]),

            html.Div(style=COL, children=[
                html.H3("✨ Enhanced", style={"color": "#76e4f7", "margin": 0, "fontSize": "15px"}),
                dcc.Loading(type="dot", color="#76e4f7", children=
                dcc.Graph(id="enhanced-spec", figure=empty_figure(),
                          config={"displayModeBar": False})),
                html.Audio(id="enhanced-audio", src="", controls=True,
                           style={"width": "100%", "marginTop": "8px"}),
                html.Div(id="vad-badge",
                         style={"color": "#76e4f7", "fontSize": "11px", "marginTop": "4px"}),
            ]),
        ]),

        html.Div(id="model-status",
                 style={"textAlign": "center", "color": "#4a5568",
                        "fontSize": "11px", "marginTop": "12px"}),

        dcc.Interval(id="probe-interval", interval=2000, n_intervals=0, max_intervals=20),

        html.P("Powered by ai-coustics SDK · Built with Dash",
               style={"textAlign": "center", "color": "#4a5568",
                      "fontSize": "11px", "marginTop": "8px"}),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Output("model-status",   "children"),
    Input("probe-interval",  "n_intervals"),
    State("model-dropdown",  "value"),
)
def refresh_models(n, current_value):
    opts    = [{"label": m, "value": m} for m in available_models]
    value   = current_value if current_value in available_models else available_models[0]
    n_found = len(available_models)
    n_total = len(ALL_MODELS)
    status  = (
        f"Models available: {n_found}/{n_total} · probing…"
        if n_found < n_total
        else f"Models available: {n_found}/{n_total}"
    )
    return opts, value, status


@app.callback(
    Output("noisy-spec",     "figure"),
    Output("noisy-audio",    "src"),
    Output("enhanced-spec",  "figure"),
    Output("enhanced-audio", "src"),
    Output("vad-badge",      "children"),
    Output("status-text",    "children"),
    Input("apply-btn",       "n_clicks"),
    State("noise-slider",           "value"),
    State("noise-type-dropdown",    "value"),
    State("enhancement-slider",     "value"),
    State("model-dropdown",         "value"),
    State("bypass-slider",          "value"),
    State("vad-sensitivity-slider", "value"),
    State("vad-hold-slider",        "value"),
    State("vad-min-slider",         "value"),
    prevent_initial_call=True,
)
def apply_all(n_clicks, noise_dbfs, noise_types_raw,
              enhancement_level, model_id,
              bypass, vad_sensitivity, vad_hold, vad_min_speech):

    noise_types = tuple(sorted(noise_types_raw or ["White"]))
    label       = " + ".join(noise_types)

    noisy         = get_noisy_audio(float(noise_dbfs), noise_types)
    f_n, t_n, S_n = compute_spectrogram(noisy, _sr)
    fig_noisy     = make_spectrogram_figure(f_n, t_n, S_n, f"{label} @ {noise_dbfs} dBFS")
    noisy_b64     = audio_to_base64_wav(noisy, _sr)

    try:
        enhanced, vad_curve = enhance(
            noisy, _sr,
            model_id=model_id,
            enhancement_level=float(enhancement_level),
            bypass=float(bypass),
            vad_sensitivity=float(vad_sensitivity),
            vad_hold=float(vad_hold),
            vad_min_speech=float(vad_min_speech),
        )
        error_msg = None
    except RuntimeError as exc:
        enhanced, vad_curve = noisy.copy(), None
        error_msg = str(exc)

    f_e, t_e, S_e = compute_spectrogram(enhanced, _sr)
    fig_enhanced  = make_spectrogram_figure(
        f_e, t_e, S_e,
        f"Enhanced — level {enhancement_level} · {model_id}",
        vad_curve=vad_curve,
    )
    enhanced_b64  = audio_to_base64_wav(enhanced, _sr)

    if error_msg:
        vad_status = f"⚠ {error_msg}"
        status     = f"Error loading model: {model_id}"
    else:
        vad_status = "● VAD overlay active" if vad_curve is not None else "VAD not available"
        status     = (
            f"Applied · {label} {noise_dbfs} dBFS · "
            f"enhancement {enhancement_level} · bypass {bypass} · {model_id}"
        )

    return fig_noisy, noisy_b64, fig_enhanced, enhanced_b64, vad_status, status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)