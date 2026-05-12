"""
Interactive outlier inspector for FlipPhone IMU recordings.

Usage:
    python inspect_outliers.py
    python inspect_outliers.py --data data/dataset.csv --method umap
    python inspect_outliers.py --method pca --port 8051

Controls:
    - Box/Lasso select points in the 3D plot to mark them as outliers.
    - The table below shows the selected recordings.
    - Click "Delete Selected from CSV" to remove them and save a cleaned CSV.
"""

import argparse
import os

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html

from train import DATA_PATH, embed_3d, extract_features, load_features

AUGMENTED_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset_augmented.csv")
DEFAULT_DATA_PATH = AUGMENTED_PATH if os.path.exists(AUGMENTED_PATH) else DATA_PATH

# ── Color map ─────────────────────────────────────────────────────────

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]


def trick_colors(tricks: list[str]) -> dict[str, str]:
    unique = sorted(set(tricks))
    return {t: COLORS[i % len(COLORS)] for i, t in enumerate(unique)}


# ── App ───────────────────────────────────────────────────────────────


def build_app(feat_df: pd.DataFrame, feature_cols: list[str], coords: np.ndarray, data_path: str) -> Dash:
    color_map = trick_colors(feat_df["trick"].tolist())
    point_colors = [color_map[t] for t in feat_df["trick"]]

    # Build scatter data per trick so the legend shows trick names
    trick_list = sorted(feat_df["trick"].unique())

    app = Dash(__name__, suppress_callback_exceptions=True)

    has_source = "source" in feat_df.columns

    def make_scatter_traces(mask: np.ndarray | None = None):
        """Build one trace per trick. mask: bool array of visible rows."""
        traces = []
        for trick in trick_list:
            idx = feat_df["trick"] == trick
            if mask is not None:
                idx = idx & mask
            sub = feat_df[idx]
            c = coords[idx.values]
            cd_cols = ["id", "collector", "timestamp"] + (["source"] if has_source else [])
            traces.append(
                dict(
                    type="scatter3d",
                    x=c[:, 0].tolist(),
                    y=c[:, 1].tolist(),
                    z=c[:, 2].tolist(),
                    mode="markers",
                    name=trick,
                    marker=dict(
                        size=5,
                        color=color_map[trick],
                        opacity=0.8,
                        symbol=[
                            "diamond" if (has_source and row == "synthetic") else "circle"
                            for row in (sub["source"].tolist() if has_source else [None] * len(sub))
                        ],
                    ),
                    customdata=sub[cd_cols].values.tolist(),
                    hovertemplate=(
                        "<b>%{meta}</b><br>"
                        "ID: %{customdata[0]}<br>"
                        "Collector: %{customdata[1]}<br>"
                        "Time: %{customdata[2]}<br>"
                        + ("Source: %{customdata[3]}<extra></extra>" if has_source else "<extra></extra>")
                    ),
                    meta=trick,
                )
            )
        return traces

    layout_3d = dict(
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#eee"),
        scene=dict(
            bgcolor="#1a1a2e",
            xaxis=dict(showgrid=True, gridcolor="#333", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#333", zeroline=False),
            zaxis=dict(showgrid=True, gridcolor="#333", zeroline=False),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#555", borderwidth=1),
        dragmode="select",
        uirevision="static",
    )

    all_tricks = sorted(feat_df["trick"].unique())
    all_collectors = sorted(feat_df["collector"].unique())
    all_sources = sorted(feat_df["source"].unique()) if has_source else []

    app.layout = html.Div(
        style={"fontFamily": "monospace", "background": "#12121f", "color": "#ddd", "minHeight": "100vh", "padding": "20px"},
        children=[
            html.H2("FlipPhone — Outlier Inspector", style={"color": "#7eb8ff", "marginBottom": "4px"}),
            html.P(
                f"{len(feat_df)} recordings · {data_path}",
                style={"color": "#666", "fontSize": "12px", "marginTop": 0},
            ),

            # ── Filters ──
            html.Div(
                style={"display": "flex", "gap": "40px", "marginBottom": "12px", "flexWrap": "wrap"},
                children=[
                    html.Div([
                        html.Label("Tricks", style={"fontSize": "12px", "color": "#aaa"}),
                        dcc.Checklist(
                            id="filter-tricks",
                            options=[{"label": t, "value": t} for t in all_tricks],
                            value=all_tricks,
                            labelStyle={"display": "block", "fontSize": "13px"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Collectors", style={"fontSize": "12px", "color": "#aaa"}),
                        dcc.Checklist(
                            id="filter-collectors",
                            options=[{"label": c, "value": c} for c in all_collectors],
                            value=all_collectors,
                            labelStyle={"display": "block", "fontSize": "13px"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ]),
                ] + ([
                    html.Div([
                        html.Label("Source", style={"fontSize": "12px", "color": "#aaa"}),
                        dcc.Checklist(
                            id="filter-sources",
                            options=[{"label": s, "value": s} for s in all_sources],
                            value=all_sources,
                            labelStyle={"display": "block", "fontSize": "13px"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ]),
                ] if has_source else []) + [
                ],
            ),

            # ── 3D Plot ──
            dcc.Graph(
                id="scatter-3d",
                figure={"data": make_scatter_traces(), "layout": layout_3d},
                config={"scrollZoom": True},
            ),

            html.P(
                "Click a point to select/deselect it. Box-select or Lasso-select to mark multiple points.",
                style={"color": "#888", "fontSize": "12px"},
            ),

            # ── Selection table ──
            html.Div(id="selection-info", style={"color": "#aaa", "fontSize": "13px", "marginBottom": "8px"}),

            dash_table.DataTable(
                id="selected-table",
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Trick", "id": "trick"},
                    {"name": "Collector", "id": "collector"},
                    {"name": "Timestamp", "id": "timestamp"},
                ] + ([{"name": "Source", "id": "source"}] if has_source else []),
                data=[],
                style_table={"overflowX": "auto", "marginBottom": "12px"},
                style_cell={"backgroundColor": "#1e1e30", "color": "#ccc", "border": "1px solid #333", "fontSize": "12px", "padding": "6px"},
                style_header={"backgroundColor": "#2a2a45", "color": "#7eb8ff", "fontWeight": "bold"},
                page_size=20,
            ),

            # ── Action buttons ──
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center"},
                children=[
                    html.Button(
                        "Delete Selected from CSV",
                        id="btn-delete",
                        n_clicks=0,
                        style={
                            "background": "#c0392b", "color": "#fff", "border": "none",
                            "padding": "10px 20px", "cursor": "pointer", "borderRadius": "4px",
                            "fontSize": "13px", "fontFamily": "monospace",
                        },
                    ),
                    html.Button(
                        "Clear Selection",
                        id="btn-clear",
                        n_clicks=0,
                        style={
                            "background": "#2c3e50", "color": "#aaa", "border": "1px solid #555",
                            "padding": "10px 20px", "cursor": "pointer", "borderRadius": "4px",
                            "fontSize": "13px", "fontFamily": "monospace",
                        },
                    ),
                    html.Div(id="status-msg", style={"color": "#7eb8ff", "fontSize": "13px"}),
                ],
            ),

            # hidden state: selected IDs
            dcc.Store(id="selected-ids", data=[]),
            # dummy component so the callback can always reference filter-sources
            html.Div(
                dcc.Checklist(id="filter-sources", options=[], value=[]),
                style={"display": "none"},
            ) if not has_source else html.Span(),
        ],
    )

    # ── Callbacks ──

    @app.callback(
        Output("scatter-3d", "figure"),
        Input("filter-tricks", "value"),
        Input("filter-collectors", "value"),
        Input("filter-sources", "value"),
    )
    def update_filter(tricks, collectors, sources):
        mask = feat_df["trick"].isin(tricks) & feat_df["collector"].isin(collectors)
        if has_source and sources:
            mask = mask & feat_df["source"].isin(sources)
        return {"data": make_scatter_traces(mask), "layout": layout_3d}

    @app.callback(
        Output("selected-ids", "data"),
        Output("btn-clear", "n_clicks"),
        Input("scatter-3d", "selectedData"),
        Input("scatter-3d", "clickData"),
        Input("btn-clear", "n_clicks"),
        State("selected-ids", "data"),
    )
    def update_selection(selected_data, click_data, clear_clicks, current_ids):
        triggered = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

        if "btn-clear" in triggered and clear_clicks:
            return [], 0

        if "clickData" in triggered:
            if not click_data or not click_data.get("points"):
                return current_ids, 0
            cd = click_data["points"][0].get("customdata")
            if not cd:
                return current_ids, 0
            clicked_id = cd[0]
            # toggle: deselect if already in list, otherwise add
            if clicked_id in current_ids:
                return [i for i in current_ids if i != clicked_id], 0
            return current_ids + [clicked_id], 0

        if not selected_data or not selected_data.get("points"):
            return current_ids, 0

        new_ids = [
            pt["customdata"][0]
            for pt in selected_data["points"]
            if pt.get("customdata")
        ]
        merged = list(dict.fromkeys(current_ids + new_ids))
        return merged, 0

    @app.callback(
        Output("selected-table", "data"),
        Output("selection-info", "children"),
        Input("selected-ids", "data"),
    )
    def show_selected(ids):
        if not ids:
            return [], "No recordings selected."
        cols = ["id", "trick", "collector", "timestamp"] + (["source"] if has_source else [])
        sub = feat_df[feat_df["id"].isin(ids)][cols]
        counts = sub["trick"].value_counts().to_dict()
        summary = " · ".join(f"{t}: {n}" for t, n in sorted(counts.items()))
        return sub.to_dict("records"), f"{len(sub)} selected — {summary}"

    @app.callback(
        Output("status-msg", "children"),
        Output("selected-ids", "data", allow_duplicate=True),
        Input("btn-delete", "n_clicks"),
        State("selected-ids", "data"),
        prevent_initial_call=True,
    )
    def delete_selected(n_clicks, ids_to_remove):
        if not ids_to_remove:
            return "Nothing selected.", ids_to_remove

        raw = pd.read_csv(data_path)
        before = raw["id"].nunique()
        cleaned = raw[~raw["id"].isin(ids_to_remove)]
        after = cleaned["id"].nunique()
        cleaned.to_csv(data_path, index=False)

        msg = f"Deleted {before - after} recording(s). CSV saved. Restart to refresh the view."
        return msg, []

    return app


# ── Entry point ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Inspect and remove outliers interactively")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="CSV dataset path")
    parser.add_argument("--method", choices=["umap", "pca"], default="umap",
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading features from {args.data} …")
    feat_df, feature_cols = load_features(
        data_path=args.data,
        selected_tricks=None,       # show all tricks for outlier inspection
        selected_collectors=None,   # show all collectors
    )

    # attach source column if present in the raw CSV
    raw = pd.read_csv(args.data)
    if "source" in raw.columns:
        source_map = raw.groupby("id")["source"].first()
        feat_df["source"] = feat_df["id"].map(source_map)

    print(f"  {len(feat_df)} recordings, {len(feature_cols)} features")

    print(f"Computing 3D embedding ({args.method.upper()}) …")
    X = feat_df[feature_cols].values
    coords = embed_3d(X, args.method, args.seed)
    print("  Done.")

    app = build_app(feat_df, feature_cols, coords, args.data)
    print(f"\nOpen http://localhost:{args.port} in your browser.\n")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
