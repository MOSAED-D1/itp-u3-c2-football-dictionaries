import io
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Football Team Analytics"


def configure_page() -> None:
	st.set_page_config(page_title=APP_TITLE, page_icon="⚽", layout="wide")
	st.title(APP_TITLE)
	st.caption("Upload match or training tracking data to analyze player performance.")


@st.cache_data(show_spinner=False)
def load_csv(file_like: io.BytesIO | str) -> pd.DataFrame:
	if isinstance(file_like, (str, os.PathLike)):
		return pd.read_csv(file_like)
	return pd.read_csv(file_like)


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure expected columns exist and types are reasonable.

	Expected columns (case-insensitive allowed):
	- player_id, player_name
	- timestamp (ISO or epoch seconds) or minute (int)
	- distance_m, speed_mps, accel_mps2
	- heart_rate
	- pass_attempted, pass_completed
	- shot_xg
	- x, y (0-100 pitch coordinates)
	- sprint (bool or 0/1)
	"""
	# Normalize columns
	df = df.copy()
	df.columns = [str(c).strip().lower() for c in df.columns]

	# Timestamp handling
	if "timestamp" in df.columns:
		try:
			df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
		except Exception:
			pass
	elif "minute" in df.columns:
		# create synthetic timestamp from minute if not present
		start = pd.Timestamp.utcnow().floor("min")
		df["timestamp"] = start + pd.to_timedelta(df["minute"].fillna(0).astype(float), unit="m")
	else:
		# if neither exists, create index-based minutes
		df["minute"] = np.arange(len(df))
		start = pd.Timestamp.utcnow().floor("min")
		df["timestamp"] = start + pd.to_timedelta(df["minute"], unit="m")

	# Numeric coercions
	for col in [
		"distance_m",
		"speed_mps",
		"accel_mps2",
		"heart_rate",
		"pass_attempted",
		"pass_completed",
		"shot_xg",
		"x",
		"y",
	]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	# Sprint boolean
	if "sprint" in df.columns:
		if df["sprint"].dtype != bool:
			df["sprint"] = df["sprint"].fillna(0).astype(int).astype(bool)
	else:
		# infer sprint by speed threshold (>=7 m/s ~ 25.2 km/h)
		if "speed_mps" in df.columns:
			df["sprint"] = (df["speed_mps"].fillna(0) >= 7.0)
		else:
			df["sprint"] = False

	# Player identity fallbacks
	if "player_name" not in df.columns:
		df["player_name"] = df.get("player_id", pd.Series(["Unknown"] * len(df)))
	if "player_id" not in df.columns:
		df["player_id"] = pd.factorize(df["player_name"])[0]

	# Minutes
	if "minute" not in df.columns:
		df["minute"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 60).astype(int)

	# Clamp pitch to [0, 100]
	for col in ["x", "y"]:
		if col in df.columns:
			df[col] = df[col].clip(0, 100)

	return df


def compute_player_kpis(df: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate KPIs per player."""
	if df.empty:
		return pd.DataFrame()
	group_cols = ["player_id", "player_name"]
	agg = df.groupby(group_cols).agg(
		minutes=("minute", lambda s: s.max() - s.min() + 1),
		total_distance_m=("distance_m", "sum"),
		avg_speed_mps=("speed_mps", "mean"),
		max_speed_mps=("speed_mps", "max"),
		sprints=("sprint", "sum"),
		heart_rate_avg=("heart_rate", "mean"),
		heart_rate_max=("heart_rate", "max"),
		passes_attempted=("pass_attempted", "sum"),
		passes_completed=("pass_completed", "sum"),
		xg_total=("shot_xg", "sum"),
	).reset_index()

	# Derived
	agg["distance_km"] = agg["total_distance_m"].fillna(0) / 1000.0
	agg["pass_completion_%"] = np.where(
		agg["passes_attempted"].fillna(0) > 0,
		agg["passes_completed"].fillna(0) / agg["passes_attempted"].clip(lower=1) * 100.0,
		np.nan,
	)
	return agg


def make_time_series(df_player: pd.DataFrame) -> go.Figure:
	fig = go.Figure()
	if "speed_mps" in df_player.columns:
		fig.add_trace(
			go.Scatter(
				x=df_player["timestamp"], y=df_player["speed_mps"], name="Speed (m/s)", mode="lines"
			)
		)
	if "heart_rate" in df_player.columns:
		fig.add_trace(
			go.Scatter(
				x=df_player["timestamp"], y=df_player["heart_rate"], name="Heart Rate", yaxis="y2", mode="lines"
			)
		)
	fig.update_layout(
		title="Time Series: Speed and Heart Rate",
		yaxis=dict(title="Speed (m/s)"),
		yaxis2=dict(title="Heart Rate", overlaying="y", side="right"),
		legend=dict(orientation="h"),
		margin=dict(l=10, r=10, t=40, b=10),
	)
	return fig


def make_radar(kpis_row: pd.Series) -> go.Figure:
	metrics = [
		("distance_km", "Distance (km)"),
		("avg_speed_mps", "Avg Speed"),
		("max_speed_mps", "Max Speed"),
		("sprints", "Sprints"),
		("pass_completion_%", "Pass %"),
		("xg_total", "xG"),
	]
	values = [float(kpis_row.get(m, np.nan) or 0) for m, _ in metrics]
	labels = [label for _, label in metrics]
	# Normalize for display if all zeros
	if np.nansum(values) == 0:
		values = [0.0001] * len(values)
	fig = go.Figure(
		go.Scatterpolar(r=values, theta=labels, fill="toself", name=str(kpis_row.get("player_name", "Player")))
	)
	fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), title="Performance Radar")
	return fig


def make_pitch_heatmap(df_player: pd.DataFrame) -> go.Figure:
	if not {"x", "y"}.issubset(df_player.columns):
		return go.Figure()
	fig = px.density_heatmap(
		df_player,
		x="x",
		y="y",
		nbinsx=30,
		nbinsy=20,
		color_continuous_scale="Viridis",
		labels={"x": "Pitch X", "y": "Pitch Y"},
	)
	fig.update_layout(
		title="Pitch Heatmap",
		xaxis=dict(range=[0, 100], constrain="domain"),
		yaxis=dict(range=[0, 100], scaleanchor="x", scaleratio=1),
		margin=dict(l=10, r=10, t=40, b=10),
		height=500,
	)
	return fig


def sidebar_controls(players: List[str]) -> Tuple[List[str], Tuple[pd.Timestamp, pd.Timestamp]]:
	with st.sidebar:
		st.header("Controls")
		selected_players = st.multiselect("Players", options=players, default=players[:1] if players else [])
		st.divider()
		st.caption("Filter by time range")
		start_default = pd.Timestamp("2022-01-01")
		end_default = pd.Timestamp("2022-01-02")
		start_ts, end_ts = st.slider(
			"Time window",
			min_value=start_default,
			max_value=end_default,
			value=(start_default, end_default),
			format="MM/DD/YY - hh:mm",
		)
	return selected_players, (start_ts, end_ts)


def layout_kpis(kpis: pd.DataFrame, selected_players: List[str]) -> None:
	st.subheader("Key Performance Indicators")
	if kpis.empty:
		st.info("No KPIs to display.")
		return
	if selected_players:
		kpis = kpis[kpis["player_name"].isin(selected_players)]
	st.dataframe(
		kpis[[
			"player_name",
			"minutes",
			"distance_km",
			"avg_speed_mps",
			"max_speed_mps",
			"sprints",
			"pass_completion_%",
			"xg_total",
		]].round({"distance_km": 2, "avg_speed_mps": 2, "max_speed_mps": 2, "pass_completion_%": 1, "xg_total": 2})
		.set_index("player_name"),
		use_container_width=True,
	)


def main() -> None:
	configure_page()

	uploaded = st.file_uploader("Upload CSV", type=["csv"])
	col_sample, col_clear = st.columns([1, 1])
	with col_sample:
		if st.button("Load sample dataset"):
			st.session_state["use_sample"] = True
	with col_clear:
		if st.button("Clear data"):
			st.session_state.pop("use_sample", None)

	df: pd.DataFrame | None = None
	if uploaded is not None:
		df = load_csv(uploaded)
	elif st.session_state.get("use_sample"):
		try:
			df = load_csv(os.path.join(os.path.dirname(__file__), "sample_data", "sample_events.csv"))
		except Exception as e:
			st.error(f"Failed to load sample data: {e}")

	if df is None:
		st.info("Upload a CSV or click 'Load sample dataset' to get started.")
		st.stop()

	# Validate and coerce schema
	df = coerce_schema(df)

	# Derive absolute min/max timestamps for slider bounds
	min_ts = pd.to_datetime(df["timestamp"]).min()
	max_ts = pd.to_datetime(df["timestamp"]).max()

	players = sorted(df["player_name"].dropna().astype(str).unique().tolist())
	selected_players, (start_ts, end_ts) = sidebar_controls(players)

	# Apply filters
	mask_time = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
	mask_player = df["player_name"].isin(selected_players) if selected_players else pd.Series(True, index=df.index)
	df_filtered = df[mask_time & mask_player].copy()

	# KPIs
	kpis = compute_player_kpis(df_filtered)
	layout_kpis(kpis, selected_players)

	# Visuals
	st.subheader("Visualizations")
	tabs = st.tabs(["Time Series", "Radar", "Pitch Heatmap"]) 

	with tabs[0]:
		if selected_players:
			for player in selected_players:
				df_p = df_filtered[df_filtered["player_name"] == player].sort_values("timestamp")
				st.plotly_chart(make_time_series(df_p), use_container_width=True)
		else:
			st.info("Select at least one player to view time series.")

	with tabs[1]:
		if not kpis.empty:
			for _, row in kpis.iterrows():
				st.plotly_chart(make_radar(row), use_container_width=True)
		else:
			st.info("No KPIs available for radar chart.")

	with tabs[2]:
		if selected_players:
			for player in selected_players:
				df_p = df_filtered[df_filtered["player_name"] == player]
				st.plotly_chart(make_pitch_heatmap(df_p), use_container_width=True)
		else:
			st.info("Select at least one player to view the heatmap.")


if __name__ == "__main__":
	main()

