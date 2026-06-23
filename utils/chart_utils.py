"""Piyasa verileri için tekrar kullanılabilir Plotly grafik araçları."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _empty_figure(title: str, message: str) -> go.Figure:
    """Boş veri durumunda kullanıcıya açıklama gösteren grafik üretir."""
    figure = go.Figure()
    figure.update_layout(
        title=title,
        autosize=True,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
    )
    return figure


def _has_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    """Verinin boş olmadığını ve istenen sütunları içerdiğini denetler."""
    return (
        isinstance(dataframe, pd.DataFrame)
        and not dataframe.empty
        and all(column in dataframe.columns for column in columns)
    )


def create_price_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Kapanış fiyatını ve hareketli ortalamaları gösteren grafik oluşturur."""
    title = "AAPL Kapanış Fiyatı ve Hareketli Ortalamalar"
    if not _has_columns(dataframe, ("Date", "Close")):
        return _empty_figure(title, "Fiyat grafiği için kullanılabilir veri yok.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dataframe["Date"],
            y=dataframe["Close"],
            mode="lines",
            name="Kapanış",
        )
    )

    for column, label in (("MA_20", "20 Günlük Ortalama"), ("MA_50", "50 Günlük Ortalama")):
        if column in dataframe.columns:
            chart_data = dataframe[["Date", column]].dropna()
            if not chart_data.empty:
                figure.add_trace(
                    go.Scatter(
                        x=chart_data["Date"],
                        y=chart_data[column],
                        mode="lines",
                        name=label,
                    )
                )

    figure.update_layout(
        title=title,
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD)",
        hovermode="x unified",
        autosize=True,
    )
    return figure


def create_volume_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Günlük işlem hacmini gösteren sütun grafik oluşturur."""
    title = "AAPL Günlük İşlem Hacmi"
    if not _has_columns(dataframe, ("Date", "Volume")):
        return _empty_figure(title, "Hacim grafiği için kullanılabilir veri yok.")

    chart_data = dataframe[["Date", "Volume"]].dropna()
    if chart_data.empty:
        return _empty_figure(title, "Hacim grafiği için kullanılabilir veri yok.")

    figure = go.Figure(
        data=[
            go.Bar(
                x=chart_data["Date"],
                y=chart_data["Volume"],
                name="İşlem Hacmi",
            )
        ]
    )
    figure.update_layout(
        title=title,
        xaxis_title="Tarih",
        yaxis_title="İşlem Hacmi",
        hovermode="x unified",
        autosize=True,
    )
    return figure


def create_return_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Günlük log-getirileri gösteren grafik oluşturur."""
    title = "AAPL Günlük Log-Getirileri"
    if not _has_columns(dataframe, ("Date", "Log_Return")):
        return _empty_figure(title, "Getiri grafiği için kullanılabilir veri yok.")

    chart_data = dataframe[["Date", "Log_Return"]].dropna()
    if chart_data.empty:
        return _empty_figure(title, "Getiri grafiği için kullanılabilir veri yok.")

    figure = go.Figure(
        data=[
            go.Scatter(
                x=chart_data["Date"],
                y=chart_data["Log_Return"],
                mode="lines",
                name="Log-Getiri",
            )
        ]
    )
    figure.update_layout(
        title=title,
        xaxis_title="Tarih",
        yaxis_title="Log-Getiri",
        hovermode="x unified",
        autosize=True,
    )
    return figure


def create_volatility_chart(dataframe: pd.DataFrame) -> go.Figure:
    """Yirmi günlük hareketli volatiliteyi gösteren grafik oluşturur."""
    title = "AAPL 20 Günlük Hareketli Volatilitesi"
    if not _has_columns(dataframe, ("Date", "Volatility_20")):
        return _empty_figure(
            title, "Volatilite grafiği için kullanılabilir veri yok."
        )

    chart_data = dataframe[["Date", "Volatility_20"]].dropna()
    if chart_data.empty:
        return _empty_figure(
            title,
            "Volatilite için yeterli sayıda gözlem bulunmuyor.",
        )

    figure = go.Figure(
        data=[
            go.Scatter(
                x=chart_data["Date"],
                y=chart_data["Volatility_20"],
                mode="lines",
                name="20 Günlük Volatilite",
            )
        ]
    )
    figure.update_layout(
        title=title,
        xaxis_title="Tarih",
        yaxis_title="Volatilite",
        hovermode="x unified",
        autosize=True,
    )
    return figure
