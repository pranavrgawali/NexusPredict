"""
NexusPredict — Plotly Chart Factory
Unified theme-aware chart helpers with consistent styling.
Supports both dark and light themes via the `theme` parameter.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Theme Color Configs ──
DARK_COLORS = {
    "indigo": "#6366f1",
    "indigo_light": "#818cf8",
    "cyan": "#22d3ee",
    "emerald": "#10b981",
    "rose": "#f43f5e",
    "amber": "#f59e0b",
    "violet": "#8b5cf6",
    "pink": "#ec4899",
    "slate": "#64748b",
    "text": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "grid": "rgba(148, 163, 184, 0.08)",
    "bg": "rgba(0,0,0,0)",
    "hover_bg": "#1e1b4b",
    "donut_border": "#0a0e1a",
}

LIGHT_COLORS = {
    "indigo": "#6366f1",
    "indigo_light": "#818cf8",
    "cyan": "#0891b2",
    "emerald": "#059669",
    "rose": "#e11d48",
    "amber": "#d97706",
    "violet": "#7c3aed",
    "pink": "#db2777",
    "slate": "#64748b",
    "text": "#0f172a",
    "text_secondary": "#475569",
    "text_muted": "#94a3b8",
    "grid": "rgba(0, 0, 0, 0.06)",
    "bg": "rgba(0,0,0,0)",
    "hover_bg": "#f1f5f9",
    "donut_border": "#ffffff",
}


def get_theme_colors(theme="dark"):
    """Return the appropriate color dictionary for the given theme."""
    return LIGHT_COLORS if theme == "light" else DARK_COLORS


COLOR_SEQUENCE = [
    "#6366f1", "#22d3ee", "#10b981", "#f59e0b",
    "#f43f5e", "#8b5cf6", "#ec4899", "#06b6d4",
]


def _get_layout_defaults(theme="dark"):
    """Generate layout defaults for the given theme (no legend — set per chart)."""
    colors = get_theme_colors(theme)
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(
            bgcolor=colors["hover_bg"],
            font_size=13,
            font_family="Inter, sans-serif",
            font_color=colors["text"],
            bordercolor="rgba(99, 102, 241, 0.3)",
        ),
    )


# Keep a default reference for backward compatibility
COLORS = DARK_COLORS
LAYOUT_DEFAULTS = _get_layout_defaults("dark")


def _apply_axes(fig, showgrid_x=False, showgrid_y=True, theme="dark"):
    """Apply consistent axis styling."""
    colors = get_theme_colors(theme)
    fig.update_xaxes(
        showgrid=showgrid_x,
        gridcolor=colors["grid"],
        linecolor="rgba(148, 163, 184, 0.1)" if theme == "dark" else "rgba(0,0,0,0.08)",
        tickfont=dict(color=colors["text_muted"], size=11),
        title_font=dict(color=colors["text_secondary"], size=12),
    )
    fig.update_yaxes(
        showgrid=showgrid_y,
        gridcolor=colors["grid"],
        linecolor="rgba(148, 163, 184, 0.1)" if theme == "dark" else "rgba(0,0,0,0.08)",
        tickfont=dict(color=colors["text_muted"], size=11),
        title_font=dict(color=colors["text_secondary"], size=12),
        zeroline=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════
# Chart Builders
# ═══════════════════════════════════════════════════════════

def create_area_chart(df, x, y, title="", y2=None, y2_name="Target",
                      color=None, height=380, theme="dark"):
    """Area chart with optional secondary line (e.g. target)."""
    colors = get_theme_colors(theme)
    if color is None:
        color = colors["indigo"]

    fig = go.Figure()

    # Main area
    fill_color = "rgba(99, 102, 241, 0.12)" if theme == "dark" else "rgba(99, 102, 241, 0.08)"
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], name=y,
        fill="tozeroy",
        fillcolor=fill_color,
        line=dict(color=color, width=2.5, shape="spline"),
        mode="lines",
        hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{y}: ₹%{{y:,.0f}}<extra></extra>",
    ))

    # Optional second line
    if y2 and y2 in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y2], name=y2_name,
            line=dict(color=colors["cyan"], width=2, dash="dot"),
            mode="lines",
            hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{y2_name}: ₹%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        height=height,
        showlegend=bool(y2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=colors["text_secondary"], size=11)),
    )
    return _apply_axes(fig, theme=theme)


def create_bar_chart(df, x, y, title="", orientation="v", color=None,
                     color_discrete_sequence=None, height=380, text_auto=False,
                     theme="dark"):
    """Vertical or horizontal bar chart."""
    colors = get_theme_colors(theme)
    if color_discrete_sequence is None:
        color_discrete_sequence = COLOR_SEQUENCE

    if orientation == "h":
        fig = go.Figure(go.Bar(
            y=df[x], x=df[y],
            orientation="h",
            marker=dict(
                color=color_discrete_sequence[:len(df)],
                line=dict(width=0),
                cornerradius=6,
            ),
            text=df[y].apply(lambda v: f"{v:,.0f}" if v > 1000 else str(v)) if text_auto else None,
            textposition="outside",
            textfont=dict(color=colors["text_secondary"], size=11),
            hovertemplate="<b>%{y}</b><br>Value: %{x:,.0f}<extra></extra>",
        ))
    else:
        fig = go.Figure(go.Bar(
            x=df[x], y=df[y],
            marker=dict(
                color=color or colors["indigo"],
                line=dict(width=0),
                cornerradius=6,
            ),
            text=df[y].apply(lambda v: f"₹{v:,.0f}" if v > 1000 else str(v)) if text_auto else None,
            textposition="outside",
            textfont=dict(color=colors["text_secondary"], size=11),
            hovertemplate="<b>%{x}</b><br>Value: %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        height=height,
    )
    return _apply_axes(fig, showgrid_x=(orientation == "h"), showgrid_y=(orientation == "v"), theme=theme)


def create_donut_chart(df, names, values, title="", height=380, hole=0.55,
                       theme="dark"):
    """Donut/pie chart."""
    colors = get_theme_colors(theme)
    fig = go.Figure(go.Pie(
        labels=df[names],
        values=df[values],
        hole=hole,
        marker=dict(
            colors=COLOR_SEQUENCE[:len(df)],
            line=dict(color=colors["donut_border"], width=2),
        ),
        textinfo="label+percent",
        textfont=dict(color=colors["text"], size=11),
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
        pull=[0.03] * len(df),
    ))

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        height=height,
        showlegend=True,
        legend=dict(
            orientation="v", yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            font=dict(color=colors["text_secondary"], size=11),
        ),
    )
    return fig


def create_scatter_chart(df, x, y, color_col=None, size_col=None, title="",
                          height=420, color_map=None, theme="dark"):
    """Scatter/bubble plot."""
    colors = get_theme_colors(theme)
    if color_map is None:
        color_map = {
            "Platinum": "#8b5cf6",
            "Gold": "#f59e0b",
            "Silver": "#94a3b8",
            "Bronze": "#d97706",
            "High": "#f43f5e",
            "Medium": "#f59e0b",
            "Low": "#10b981",
        }

    fig = px.scatter(
        df, x=x, y=y,
        color=color_col,
        size=size_col,
        color_discrete_map=color_map,
        hover_data=df.columns.tolist(),
        height=height,
    )

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.1)" if theme == "dark" else "rgba(0,0,0,0.1)")),
    )

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=colors["text_secondary"], size=11),
        ),
    )
    return _apply_axes(fig, theme=theme)


def create_forecast_chart(hist_df, forecast_df, title="", height=420,
                          theme="dark"):
    """Time-series forecast chart with confidence band."""
    colors = get_theme_colors(theme)
    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_df["Date"], y=hist_df["Units Sold"],
        name="Historical",
        line=dict(color=colors["indigo"], width=2),
        mode="lines",
        hovertemplate="<b>%{x|%b %d}</b><br>Actual: %{y:,.0f} units<extra></extra>",
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
        y=pd.concat([forecast_df["Upper Bound"], forecast_df["Lower Bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(34, 211, 238, 0.08)" if theme == "dark" else "rgba(34, 211, 238, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Units Sold"],
        name="Forecast",
        line=dict(color=colors["cyan"], width=2.5, dash="dot"),
        mode="lines+markers",
        marker=dict(size=4, color=colors["cyan"]),
        hovertemplate="<b>%{x|%b %d}</b><br>Forecast: %{y:,.0f} units<extra></extra>",
    ))

    # Divider line at today
    if len(hist_df) > 0 and len(forecast_df) > 0:
        today = pd.Timestamp(hist_df["Date"].iloc[-1])
        fig.add_shape(
            type="line", x0=today, x1=today, y0=0, y1=1,
            yref="paper", line=dict(dash="dash", color="rgba(148, 163, 184, 0.3)", width=1),
        )
        fig.add_annotation(
            x=today, y=1.05, yref="paper", text="Today",
            showarrow=False, font=dict(color=colors["text_muted"], size=11),
        )

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=colors["text_secondary"], size=11),
        ),
    )
    return _apply_axes(fig, theme=theme)


def create_prophet_forecast_chart(forecast_df, title="", height=440,
                                   theme="dark"):
    """
    Plotly chart for Prophet model forecast output.
    Expects columns: ds, yhat, yhat_lower, yhat_upper
    """
    colors = get_theme_colors(theme)
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.08)" if theme == "dark" else "rgba(99, 102, 241, 0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Upper bound dashed
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat_upper"],
        name="Upper Bound",
        line=dict(color="rgba(99, 102, 241, 0.3)", width=1, dash="dot"),
        mode="lines",
        showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>Upper: %{y:,.0f}<extra></extra>",
    ))

    # Lower bound dashed
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat_lower"],
        name="Lower Bound",
        line=dict(color="rgba(99, 102, 241, 0.3)", width=1, dash="dot"),
        mode="lines",
        showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>Lower: %{y:,.0f}<extra></extra>",
    ))

    # Main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        name="Predicted Demand",
        line=dict(color=colors["indigo"], width=2.5),
        mode="lines+markers",
        marker=dict(size=5, color=colors["indigo"], line=dict(width=1, color="rgba(255,255,255,0.2)")),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: %{y:,.0f} units<extra></extra>",
    ))

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=colors["text_secondary"], size=11),
        ),
        xaxis_title="Date",
        yaxis_title="Demand (Units)",
    )
    return _apply_axes(fig, theme=theme)


def create_gauge_chart(value, title="", max_val=100, height=250,
                        thresholds=None, theme="dark"):
    """Gauge/indicator chart for single metric display."""
    colors = get_theme_colors(theme)
    if thresholds is None:
        thresholds = [
            (30, colors["emerald"]),
            (60, colors["amber"]),
            (100, colors["rose"]),
        ]

    # Determine color
    bar_color = colors["indigo"]
    for thresh_val, color in thresholds:
        if value <= thresh_val:
            bar_color = color
            break

    steps = []
    prev = 0
    for thresh_val, color in thresholds:
        # Convert hex color to rgba with low opacity
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            step_color = f"rgba({r},{g},{b},0.1)"
        else:
            step_color = color.replace(")", ", 0.1)").replace("rgb(", "rgba(")
        steps.append(dict(range=[prev, thresh_val], color=step_color))
        prev = thresh_val

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title=dict(text=title, font=dict(size=14, color=colors["text_secondary"])),
        number=dict(font=dict(size=36, color=colors["text"], family="Outfit"), suffix="%"),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor=colors["text_muted"],
                      tickfont=dict(color=colors["text_muted"], size=10)),
            bar=dict(color=bar_color, thickness=0.75),
            bgcolor="rgba(255,255,255,0.03)" if theme == "dark" else "rgba(0,0,0,0.03)",
            borderwidth=0,
            steps=steps,
            threshold=dict(
                line=dict(color=colors["rose"], width=2),
                thickness=0.8,
                value=value,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"]),
        height=height,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def create_horizontal_importance_chart(df, feature_col, value_col, title="",
                                        height=350, color=None, theme="dark"):
    """Horizontal bar chart for feature importance."""
    colors = get_theme_colors(theme)
    if color is None:
        color = colors["indigo"]
    df_sorted = df.sort_values(value_col, ascending=True)

    fig = go.Figure(go.Bar(
        y=df_sorted[feature_col],
        x=df_sorted[value_col],
        orientation="h",
        marker=dict(
            color=[f"rgba(99, 102, 241, {0.3 + 0.7 * (v / df_sorted[value_col].max())})"
                   for v in df_sorted[value_col]],
            line=dict(width=0),
            cornerradius=4,
        ),
        text=df_sorted[value_col].apply(lambda v: f"{v:.2f}"),
        textposition="outside",
        textfont=dict(color=colors["text_secondary"], size=11),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        height=height,
    )
    return _apply_axes(fig, showgrid_x=True, showgrid_y=False, theme=theme)


def create_grouped_bar_chart(df, x, y_cols, names=None, title="",
                              colors_list=None, height=380, theme="dark"):
    """Grouped bar chart with multiple y columns."""
    colors = get_theme_colors(theme)
    if colors_list is None:
        colors_list = COLOR_SEQUENCE
    if names is None:
        names = y_cols

    fig = go.Figure()
    for i, (col, name) in enumerate(zip(y_cols, names)):
        fig.add_trace(go.Bar(
            x=df[x], y=df[col], name=name,
            marker=dict(color=colors_list[i % len(colors_list)], cornerradius=6),
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        **_get_layout_defaults(theme),
        title=dict(text=title, font=dict(size=15, color=colors["text"])),
        height=height,
        barmode="group",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=colors["text_secondary"], size=11),
        ),
    )
    return _apply_axes(fig, theme=theme)
