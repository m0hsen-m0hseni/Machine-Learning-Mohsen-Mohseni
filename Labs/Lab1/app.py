from dash import Dash, dcc, html, Input, Output, State, dash_table
import pandas as pd
import recommender

app = Dash(__name__)
app.title = "Movie Recommendation System"


def info_card(title, value):
    return html.Div(
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "14px",
            "borderRadius": "10px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            "flex": "1",
            "minWidth": "180px",
        },
        children=[
            html.Div(title, style={"fontSize": "13px",
                     "color": "#666", "marginBottom": "6px"}),
            html.Div(value, style={"fontSize": "18px", "fontWeight": "600"})
        ]
    )


def get_selected_movie_info(movie_title):
    return recommender.get_movie_details(movie_title)


app.layout = html.Div(
    style={
        "maxWidth": "1100px",
        "margin": "30px auto",
        "padding": "20px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#ffffff",
    },
    children=[
        html.H1(
            "Movie Recommendation System",
            style={"marginBottom": "8px", "fontSize": "42px"}
        ),
        html.P(
            "Choose a movie and get recommendations using diffrent recommendation methods",
            style={"color": "#555", "fontSize": "18px", "marginBottom": "25px"}
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 1fr 1fr",
                "gap": "14px",
                "marginBottom": "20px"
            },
            children=[
                html.Div([
                    html.Label("Select movie", style={
                               "fontWeight": "600", "marginBottom": "8px", "display": "black"}),
                    dcc.Dropdown(
                        id="movie-dropdown",
                        options=[{"label": title, "value": title}
                                 for title in recommender.movie_titles],
                        placeholder="Search and select a movie...",
                        value=recommender.movie_titles[0] if recommender.movie_titles else None,
                        style={"fontSize": "16px"}
                    ),
                ]),

                html.Div([
                    html.Label("Method", style={
                               "fontWeight": "600", "marginBottom": "8px", "display": "black"}),
                    dcc.Dropdown(
                        id="method-dropdown",
                        options=[
                            {"label": "KNN + TF-IDF", "value": "knn"},
                            {"label": "Content-Based (Cosine)",
                             "value": "content"},
                            {"label": "Genre-Based Baseline", "value": "genre"},
                        ],
                        value="knn",
                        clearable=False,
                        style={"fontSize": "16px"}
                    ),
                ]),

                html.Div([
                    html.Label("Number of recommendations", style={
                               "fontWeight": "600", "marginBottom": "8px", "display": "balck"}),
                    dcc.Slider(
                        id="topn-slider",
                        min=3,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(3, 11)},
                    ),
                ]),
            ]
        ),

        html.Div(
            style={"display": "flex", "gap": "12px", "marginBottom": "28px"},
            children=[
                html.Button(
                    "Get Recommendations",
                    id="recommend-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#1f77b4",
                        "color": "white",
                        "border": "none",
                        "padding": "12px 20px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontSize": "16px",
                        "fontWeight": "600"
                    }
                ),
                html.Button(
                    "Clear",
                    id="clear-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#e9ecef",
                        "color": "#333",
                        "border": "none",
                        "padding": "12px 20px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontSize": "16px",
                    }
                ),
            ]
        ),

        html.Div(
            id="selected-movie-info",
            style={"display": "flex", "gap": "14px",
                   "flexWrap": "wrap", "marginbottom": "28px"}
        ),

        html.Div(id="output-area")
    ]
)


@app.callback(
    Output("selected-movie-info", "children"),
    Input("movie-dropdown", "value")
)
def update_movie_info(selected_movie):
    if not selected_movie:
        return []

    info = get_selected_movie_info(selected_movie)
    tags_short = info["tags"][:120] + \
        "..." if len(str(info["tags"])) > 120 else info["tags"]

    links_row = html.Div(
        style={
            "width": "100%",
            "display": "flex",
            "gap": "16px",
            "marginTop": "6px",
            "alignItems": "center",
            "flexWrap": "wrap",
        },
        children=[
            html.A(
                "Open in IMDb",
                href=info["imdb_url"],
                target="_blank",
                style={"color": "#1f77b4", "fontWeight": "600",
                       "textDecoration": "none"},
            ) if info["imdb_url"] else html.Span("IMDb unavailable", style={"color": "#777"}),

            html.A(
                "Open in TMDb",
                href=info["tmdb_url"],
                target="_blank",
                style={"color": "#1f77b4", "fontWeight": "600",
                       "textDecoration": "none"},
            ) if info["tmdb_url"] else html.Span("TMDb unavailable", style={"color": "#777"}),
            html.Span(
                f"IMDb ID: {int(info['imdb_id']) if pd.notna(info['imdb_id']) else 'N/A'}",
                style={"color": "#555"},
            ),

            html.Span(
                f"TMDb ID: {int(info['tmdb_id']) if pd.notna(info['tmdb_id']) else 'N/A'}",
                style={"color": "#555"},
            ),
        ]
    )

    return [
        info_card("Selected Movie", info["title"]),
        info_card("Year", info["year"]),
        info_card("Genres", info["genres"]),
        info_card("Tags", tags_short),
        links_row
    ]


@app.callback(
    Output("output-area", "children"),
    Input("recommend-button", "n_clicks"),
    Input("clear-button", "n_clicks"),
    State("movie-dropdown", "value"),
    State("method-dropdown", "value"),
    State("topn-slider", "value"),
    prevent_initial_call=True
)
def update_recommendations(recommend_clisks, clear_clicks, selected_movie, selected_method, top_n):
    from dash import callback_context

    ctx = callback_context
    if not ctx.triggered:
        return html.Div()

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "clear-button":
        return html.Div()

    if not selected_movie:
        return html.Div(
            "Please select a movie first.",
            style={"color": "#b00020", "fontWeight": "600", "marginTop": "10px"}
        )

    try:

        if selected_method == "knn":
            results = recommender.recommend_by_knn(
                selected_movie,
                recommender.movies_content,
                recommender.knn_model,
                recommender.tfidf_matrix,
                top_n=top_n
            )
            method_title = "KNN + TF-IDF Recommendations"

        elif selected_method == "content":
            results = recommender.recommend_by_content(
                selected_movie,
                recommender.movies_content,
                recommender.content_similarity,
                top_n=top_n
            )
            method_title = "Content-Based Recommendations"
        else:
            results = recommender.recommend_by_genre(
                selected_movie,
                recommender.movies_genre,
                recommender.genre_similarity,
                top_n=top_n
            )
            method_title = "Genre-Based Baseline Recommendations"

        if isinstance(results, str):
            return html.Div(
                results,
                style={"color": "#b00020",
                       "fontWeight": "600", "marginTop": "10px"}
            )

        if results.empty:
            return html.Div(
                "No recommendations found.",
                style={"color": "#b00020",
                       "fontWeight": "600", "margintop": "10px"}
            )

        return html.Div([
            html.H3(method_title, style={"marginBottom": "14px"}),
            dash_table.DataTable(
                data=results.to_dict("records"),
                columns=[{"name": col.capitalize(), "id": col}
                         for col in results.columns],
                page_size=top_n,
                style_table={
                    "overflowX": "auto",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px"
                },
                style_header={
                    "backgroundColor": "f1f3f5",
                    "fontWeight": "bold",
                    "textAlign": "left"
                },
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "fontSize": "15px",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
            )
        ])

    except Exception as e:
        return html.Div(
            f"Error: {str(e)}",
            style={"color": "#b00020", "fontWeight": "600", "marginTop": "10px"}
        )


if __name__ == "__main__":
    app.run(debug=True)
