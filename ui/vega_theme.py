"""
Meridian UI - Vega Theme
Global configuration for all charts
"""

VEGA_CONFIG = {
    "background": "white",
    "padding": {"left": 10, "right": 10, "top": 6, "bottom": 6},
    "view": {"stroke": None},
    "axis": {
        "domain": False,
        "ticks": False,
        "grid": True,
        "gridColor": "#E5E7EB",
        "gridOpacity": 1,
        "labelColor": "#6B7280",
        "titleColor": "#6B7280",
        "labelFont": "Inter",
        "titleFont": "Inter",
        "labelFontSize": 11,
        "titleFontSize": 11,
        "titleFontWeight": 500
    },
    "legend": {
        "title": None,
        "labelColor": "#6B7280",
        "labelFont": "Inter",
        "labelFontSize": 11,
        "symbolType": "circle",
        "symbolSize": 80
    },
    "title": {
        "font": "Inter",
        "fontSize": 14,
        "fontWeight": 600,
        "color": "#111827",
        "anchor": "start",
        "offset": 8
    },
    "header": {
        "labelFont": "Inter",
        "labelFontSize": 12,
        "labelFontWeight": 600,
        "labelColor": "#111827"
    },
    "mark": {"color": "#111827"},
    "range": {
        "category": ["#111827", "#6B7280", "#9CA3AF", "#D1D5DB"],
        "heatmap": ["#F9FAFB", "#111827"]
    }
}


def apply_theme(spec: dict) -> dict:
    """Apply global theme to any Vega-Lite spec"""
    spec = dict(spec)
    spec["config"] = {**VEGA_CONFIG, **spec.get("config", {})}
    return spec


def ci_bar(value: float, lo: float, hi: float, n: int, title: str = "Accuracy (95% CI)") -> dict:
    """Accuracy with confidence interval - horizontal bar"""
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": 70,
        "title": title,
        "data": {"values": [{"metric": "Accuracy", "value": value, "lo": lo, "hi": hi}]},
        "layer": [
            {
                "mark": {"type": "rule", "strokeWidth": 6, "opacity": 0.18},
                "encoding": {
                    "x": {"field": "lo", "type": "quantitative", "scale": {"domain": [0, 100]}, "axis": {"title": "Accuracy (%)"}},
                    "x2": {"field": "hi"},
                    "y": {"field": "metric", "type": "nominal", "axis": None}
                }
            },
            {
                "mark": {"type": "point", "filled": True, "size": 160},
                "encoding": {
                    "x": {"field": "value", "type": "quantitative"},
                    "y": {"field": "metric", "type": "nominal"},
                    "color": {"value": "#111827"}
                }
            },
            {
                "mark": {"type": "text", "dx": 10, "dy": 0, "fontSize": 12, "fontWeight": 600, "align": "left"},
                "encoding": {
                    "x": {"field": "value", "type": "quantitative"},
                    "y": {"field": "metric", "type": "nominal"},
                    "text": {"value": f"{value:.0f}%  (n={n})"},
                    "color": {"value": "#111827"}
                }
            }
        ]
    }
    return apply_theme(spec)


def dumbbell(a: float, b: float, label_a: str = "A", label_b: str = "B", title: str = "Accuracy comparison") -> dict:
    """Compare two values with dumbbell chart"""
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": 90,
        "title": title,
        "data": {"values": [
            {"metric": "Accuracy", "model": label_a, "value": a},
            {"metric": "Accuracy", "model": label_b, "value": b}
        ]},
        "transform": [
            {"joinaggregate": [
                {"op": "max", "field": "value", "as": "maxv"},
                {"op": "min", "field": "value", "as": "minv"}
            ], "groupby": ["metric"]}
        ],
        "layer": [
            {"mark": {"type": "rule", "strokeWidth": 2, "opacity": 0.25},
             "encoding": {
                 "x": {"field": "minv", "type": "quantitative", "scale": {"domain": [0, 100]}, "axis": {"title": "Accuracy (%)"}},
                 "x2": {"field": "maxv"},
                 "y": {"field": "metric", "type": "nominal", "axis": None}
             }},
            {"mark": {"type": "point", "filled": True, "size": 160},
             "encoding": {
                 "x": {"field": "value", "type": "quantitative"},
                 "y": {"field": "metric", "type": "nominal"},
                 "color": {"field": "model", "type": "nominal",
                           "scale": {"range": ["#111827", "#6B7280"]},
                           "legend": {"orient": "right"}}
             }},
            {"mark": {"type": "text", "dy": -18, "fontSize": 12, "fontWeight": 600},
             "encoding": {
                 "x": {"field": "value", "type": "quantitative"},
                 "y": {"field": "metric", "type": "nominal"},
                 "text": {"field": "value", "type": "quantitative", "format": ".0f"},
                 "color": {"field": "model", "type": "nominal", "scale": {"range": ["#111827", "#6B7280"]}, "legend": None}
             }}
        ]
    }
    return apply_theme(spec)


def latency_boxplot(values_a: list, values_b: list, label_a: str = "A", label_b: str = "B", title: str = "Latency distribution") -> dict:
    """Boxplot comparing latency of two runs"""
    data = [{"model": label_a, "latency_s": v} for v in values_a] + \
           [{"model": label_b, "latency_s": v} for v in values_b]
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": 120,
        "title": title,
        "data": {"values": data},
        "mark": {"type": "boxplot", "extent": "min-max", "size": 18},
        "encoding": {
            "x": {"field": "latency_s", "type": "quantitative", "title": "Latency (seconds)"},
            "y": {"field": "model", "type": "nominal", "title": None},
            "color": {"field": "model", "type": "nominal", "scale": {"range": ["#111827", "#6B7280"]}, "legend": None}
        }
    }
    return apply_theme(spec)


def single_boxplot(values: list, title: str = "Latency distribution") -> dict:
    """Single boxplot for one run"""
    data = [{"metric": "Latency", "value": v} for v in values]
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": 60,
        "title": title,
        "data": {"values": data},
        "mark": {"type": "boxplot", "extent": "min-max", "size": 20, "color": "#111827"},
        "encoding": {
            "x": {"field": "value", "type": "quantitative", "title": "Latency (seconds)"},
            "y": {"field": "metric", "type": "nominal", "axis": None}
        }
    }
    return apply_theme(spec)
