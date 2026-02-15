
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RACES = [
    ("mara", 42.195),
    ("half", 21.0975),
    ("tenkm", 10.0),
    ("fivekm", 5.0),
    ("two", 3),
]

COLS = [f"{r}-netto" for r, _ in RACES] + [f"{r}-brutto" for r, _ in RACES]


def read_run26_csv_robust(csv_path: Path) -> pd.DataFrame:
    """
    run26.csv ist semikolon-separiert, aber Zeilen haben unterschiedliche Feldanzahl:
      - Header: 10 Spalten
      - Daten: häufig 9 oder 10 Felder, einmal 11 (inkl. 'Unnamed: 11')
    Wir normalisieren jede Zeile auf exakt 10 Felder.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(";")
        if len(header) != 10:
            raise ValueError(f"Unerwarteter Header mit {len(header)} Feldern: {header}")

        rows = []
        for line_no, line in enumerate(f, start=2):
            parts = line.strip().split(";")

            # Sonderfall: 11 Felder mit 'Unnamed: 11' als zusätzlichem Feld
            if len(parts) == 11 and parts[8].startswith("Unnamed"):
                parts.pop(8)

            # Auf 10 auffüllen oder kürzen
            if len(parts) < 10:
                parts = parts + [""] * (10 - len(parts))
            elif len(parts) > 10:
                parts = parts[:10]

            rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    return df


def to_seconds(series: pd.Series) -> pd.Series:
    """
    Zeitstrings wie '02:13:30' oder '00:10:26.380000' -> Sekunden (float).
    Ungültige/fehlende -> NaN.
    """
    td = pd.to_timedelta(series, errors="coerce")
    return td.dt.total_seconds()
from matplotlib.widgets import Slider

from matplotlib.widgets import Slider

def interactive_distance_distribution_with_slider_by_race(
    df_sec: pd.DataFrame,
    t_min_minutes: int = 0,
    t_max_minutes: int = 8 * 60,
    bin_width_km: float = 0.25,
    max_distance_km: float = 42.195,
    min_netto_s: int = 60,
    max_speed_kmh: float = 30.0,
    max_delay_s: int = 2 * 3600
) -> None:
    """
    Interaktiver Plot (eine Linie pro Rennen, automatische Farben):
      - Slider: Zeit seit Startschuss (0..8h)
      - x: Distanz (0..42.195 km)
      - y: Anzahl Läufer in Distanz-Bins zum Zeitpunkt T
    Delay berücksichtigt: dist(T)=0 für T<delay, danach konstante Speed aus Nettozeit.
    """

    edges = np.arange(0, max_distance_km + bin_width_km, bin_width_km)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Precompute pro Rennen: (race, race_dist_km, delay_s_array, speed_kmh_array)
    per_race = []
    for race, dist_km in RACES:
        netto = df_sec[f"{race}-netto"]
        brutto = df_sec[f"{race}-brutto"]
        delay = (brutto - netto)

        mask = (
            netto.notna() & brutto.notna() &
            (netto >= min_netto_s) & (netto <= 24 * 3600) &
            (delay >= 0) & (delay <= max_delay_s)
        )
        if mask.sum() == 0:
            continue

        netto_s = netto[mask].astype(float)
        delay_s = delay[mask].astype(float)

        speed_kmh = dist_km / (netto_s / 3600.0)
        speed_kmh = speed_kmh[(speed_kmh > 0) & (speed_kmh <= max_speed_kmh)]
        delay_s = delay_s.loc[speed_kmh.index]

        if len(speed_kmh) == 0:
            continue

        per_race.append((race, dist_km, delay_s.values, speed_kmh.values))

    if not per_race:
        raise RuntimeError("Keine verwertbaren Daten (delay/speed) für den Slider-Plot gefunden.")

    def distances_for_race_at_time(T_seconds: float, dist_km: float, delay_s: np.ndarray, speed_kmh: np.ndarray) -> np.ndarray:
        run_time_s = T_seconds - delay_s
        run_time_s = np.maximum(run_time_s, 0.0)
        d = speed_kmh * (run_time_s / 3600.0)
        return np.minimum(d, dist_km)

    def hist_counts(distances_km: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(distances_km, bins=edges)
        return counts

    # Initialzeit
    T0_min = 60
    T0_sec = T0_min * 60.0

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.18)

    lines = {}
    max0 = 0

    for race, dist_km, delay_s, speed_kmh in per_race:
        d0 = distances_for_race_at_time(T0_sec, dist_km, delay_s, speed_kmh)
        c0 = hist_counts(d0)
        (ln,) = ax.plot(centers, c0, linewidth=2, label=race)
        lines[race] = ln
        if c0.max() > max0:
            max0 = int(c0.max())

    ax.set_title(f"Distanzverteilung nach {T0_min} min (Delay berücksichtigt) – pro Rennen")
    ax.set_xlabel("Distanz (km)")
    ax.set_ylabel("Anzahl Läufer im Bin")
    ax.set_xlim(0, max_distance_km)
    ax.set_ylim(0, max(5, int(max0 * 1.05)))
    ax.grid(True, alpha=0.25)
    ax.legend(title="Rennen")

    slider_ax = fig.add_axes([0.12, 0.06, 0.76, 0.04])
    slider = Slider(
        ax=slider_ax,
        label="Zeit seit Startschuss (min)",
        valmin=t_min_minutes,
        valmax=t_max_minutes,
        valinit=T0_min,
        valstep=1
    )

    def on_change(val):
        T_min = float(val)
        T_sec = T_min * 60.0

        ymax = 0
        for race, dist_km, delay_s, speed_kmh in per_race:
            d = distances_for_race_at_time(T_sec, dist_km, delay_s, speed_kmh)
            c = hist_counts(d)
            lines[race].set_ydata(c)
            if c.max() > ymax:
                ymax = int(c.max())

        ax.set_title(f"Distanzverteilung nach {int(T_min)} min (Delay berücksichtigt) – pro Rennen")
        ax.set_ylim(0, max(5, int(ymax * 1.05)))
        fig.canvas.draw_idle()

    slider.on_changed(on_change)

    plt.show()

def plot_delay_vs_speed_scatter(df_sec: pd.DataFrame,
                                max_delay_s: int = 60 * 60,
                                min_netto_s: int = 60,
                                max_speed_kmh: float = 30.0) -> None:
    """
    Scatter: x=delay (s), y=speed (km/h)
    delay = brutto - netto
    speed = dist_km / (netto_s/3600)
    Eine Punktwolke pro Rennen.
    """

    plt.figure(figsize=(12, 6))

    plotted_any = False

    for race, dist_km in RACES:
        netto = df_sec[f"{race}-netto"]
        brutto = df_sec[f"{race}-brutto"]

        delay = (brutto - netto)
        # Plausibilität
        mask = (
            netto.notna() & brutto.notna() &
            (netto >= min_netto_s) & (netto <= 24 * 3600) &
            (delay >= 0) & (delay <= max_delay_s)
        )

        if mask.sum() == 0:
            continue

        netto_s = netto[mask].astype(float)
        delay_s = delay[mask].astype(float)

        speed_kmh = dist_km / (netto_s / 3600.0)
        speed_kmh = speed_kmh[(speed_kmh > 0) & (speed_kmh <= max_speed_kmh)]

        # Delay muss zu speed gefiltert passen
        delay_s = delay_s.loc[speed_kmh.index]

        if len(speed_kmh) == 0:
            continue

        plt.scatter(delay_s.values, speed_kmh.values, s=12, alpha=0.45, label=race)
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("Keine verwertbaren Daten für Delay-vs-Speed Scatter gefunden.")

    plt.title("Delay vs Speed (x = Startverzögerung, y = Geschwindigkeit)")
    plt.xlabel("Startverzögerung (Sekunden) = Brutto − Netto")
    plt.ylabel("Geschwindigkeit (km/h) = Distanz / Nettozeit")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Rennen")
    plt.tight_layout()
    plt.show()

def format_hms(seconds: float) -> str:
    """Sekunden -> HH:MM:SS"""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_race_statistics(df_sec: pd.DataFrame) -> None:
    print("\nRennen – Teilnehmer & Zeiten (Netto/Brutto)\n")
    print(
        f"{'Rennen':<10} "
        f"{'Teilnehmer':>12} "
        f"{'Schnellster':>14} "
        f"{'Langsamster':>14} "
        f"{'Ø Delay':>12} "
        f"{'Max Delay':>12}"
    )
    print("-" * 86)

    for race, _dist in RACES:
        # Netto-/Brutto-Spalten je Rennen (erwartete Namen wie bisher)
        netto = df_sec.get(f"{race}-netto", pd.Series(dtype=float)).dropna()
        brutto = df_sec.get(f"{race}-brutto", pd.Series(dtype=float)).dropna()

        # Nur plausible Zeiten behalten (0 .. <24h)
        netto = netto[(netto > 0) & (netto < 24 * 3600)]
        brutto = brutto[(brutto > 0) & (brutto < 24 * 3600)]

        # Für Delay-Auswertung brauchen wir Paare (nur IDs, die beide Werte haben)
        pair = pd.concat(
            [df_sec.get(f"{race}-netto"), df_sec.get(f"{race}-brutto")],
            axis=1,
            keys=["netto", "brutto"],
        ).dropna()

        if not pair.empty:
            pair = pair[(pair["netto"] > 0) & (pair["netto"] < 24 * 3600)]
            pair = pair[(pair["brutto"] > 0) & (pair["brutto"] < 24 * 3600)]
            delay = (pair["brutto"] - pair["netto"])
            delay = delay[(delay >= 0) & (delay < 24 * 3600)]
        else:
            delay = pd.Series(dtype=float)

        if netto.empty:
            print(
                f"{race:<10} "
                f"{'0':>12} "
                f"{'-':>14} "
                f"{'-':>14} "
                f"{'-':>12} "
                f"{'-':>12}"
            )
            continue

        fastest = float(netto.min())
        slowest = float(netto.max())

        # Delay-Statistiken: "wie viel später startet der letzte Starter"
        # Interpretation: max(brutto - netto) innerhalb des Rennens
        mean_delay = float(delay.mean()) if not delay.empty else None
        max_delay = float(delay.max()) if not delay.empty else None

        print(
            f"{race:<10} "
            f"{len(netto):>12} "
            f"{format_hms(fastest):>14} "
            f"{format_hms(slowest):>14} "
            f"{(format_hms(mean_delay) if mean_delay is not None else '-'):>12} "
            f"{(format_hms(max_delay) if max_delay is not None else '-'):>12}"
        )


def plot_speed_distribution_lines(df_sec: pd.DataFrame,
                                  bin_width_kmh: float = 0.25,
                                  min_netto_s: int = 60,
                                  max_speed_kmh: float = 30.0) -> None:
    """
    Verteilung der Läufer über Geschwindigkeit (km/h) als Histogramm-Linechart.
    Eine Linie pro Rennen.

    speed_kmh = dist_km / (netto_s/3600)
    """

    edges = np.arange(0, max_speed_kmh + bin_width_kmh, bin_width_kmh)
    centers = (edges[:-1] + edges[1:]) / 2.0

    plt.figure(figsize=(12, 6))
    plotted_any = False

    for race, dist_km in RACES:
        netto = df_sec[f"{race}-netto"].dropna()
        netto = netto[(netto >= min_netto_s) & (netto <= 24 * 3600)]
        if netto.empty:
            continue

        speed_kmh = dist_km / (netto.astype(float) / 3600.0)
        speed_kmh = speed_kmh[(speed_kmh > 0) & (speed_kmh <= max_speed_kmh)]
        if speed_kmh.empty:
            continue

        counts, _ = np.histogram(speed_kmh.values, bins=edges)
        plt.plot(centers, counts, linewidth=2, label=race)
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("Keine verwertbaren Daten für Speed-Verteilung gefunden.")

    plt.title(f"Geschwindigkeitsverteilung: Anzahl Läufer pro {bin_width_kmh:.2f} km/h-Bin")
    plt.xlabel("Geschwindigkeit (km/h)")
    plt.ylabel("Anzahl Läufer im Bin")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Rennen")
    plt.tight_layout()
    plt.show()
def plot_delay_histogram_by_race(
    df_sec: pd.DataFrame,
    bin_width_s: int = 10,
    max_delay_s: int = 60 * 60,
    min_netto_s: int = 60
) -> None:
    """
    Statischer Plot:
      - Histogramm über Startdelay (Sekunden) = brutto - netto
      - Farben pro Rennen (gestapelt), um Startwellen sichtbar zu machen
    """

    edges = np.arange(0, max_delay_s + bin_width_s, bin_width_s)

    delays_per_race = []
    labels = []

    for race, _dist in RACES:
        netto = df_sec[f"{race}-netto"]
        brutto = df_sec[f"{race}-brutto"]
        delay = (brutto - netto)

        mask = (
            netto.notna() & brutto.notna() &
            (netto >= min_netto_s) & (netto <= 24 * 3600) &
            (delay >= 0) & (delay <= max_delay_s)
        )

        d = delay[mask].astype(float).values
        if d.size == 0:
            continue

        delays_per_race.append(d)
        labels.append(race)

    if not delays_per_race:
        raise RuntimeError("Keine verwertbaren Delay-Daten gefunden.")

    plt.figure(figsize=(12, 6))

    # Gestapelt, damit Wellen + Race-Anteile sichtbar sind
    plt.hist(delays_per_race, bins=edges, stacked=True, alpha=0.85, label=labels)

    plt.title(f"Startwellen: Verteilung der Läufer über Startdelay (Bin = {bin_width_s}s)")
    plt.xlabel("Startdelay (Sekunden) = Brutto − Netto")
    plt.ylabel("Anzahl Läufer im Bin")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Rennen")
    plt.tight_layout()
    plt.show()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("runfinal.csv"), help="Pfad zu run26.csv")
    args = ap.parse_args()

    df = read_run26_csv_robust(args.csv)

    # In Sekunden umrechnen
    df_sec = pd.DataFrame()
    for col in df.columns:
        df_sec[col] = to_seconds(df[col])

    # Tabelle behalten
    print_race_statistics(df_sec)

    # Scatter anzeigen (kein Datei-Output)
   # plot_delay_vs_speed_scatter(df_sec)
#    plot_speed_distribution_lines(df_sec, bin_width_kmh=0.25, max_speed_kmh=30.0)
   # interactive_distance_distribution_with_slider_by_race(df_sec, t_max_minutes=8*60, bin_width_km=0.25)
    plot_delay_histogram_by_race(df_sec, bin_width_s=10, max_delay_s=60*60)

if __name__ == "__main__":
    main()

