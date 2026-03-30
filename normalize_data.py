"""
Dataset Normalizer for TSP/VRPTW instances
===========================================
Convertit tous les fichiers de dataset_raw (ou dataset_split) vers
un format CSV unifié et normalisé.

Format CSV de sortie (colonnes standardisées) :
─────────────────────────────────────────────────────────────────
  source          : nom du fichier source
  instance_id     : identifiant unique de l'instance (ex: C101)
  category        : catégorie (C1, R2, RC1, tsptw, tsp, world)
  node_id         : identifiant du nœud (0 = dépôt pour VRPTW)
  x               : coordonnée X (normalisée entre 0 et 1)
  y               : coordonnée Y (normalisée entre 0 et 1)
  demand          : demande du client (0 si non applicable)
  ready_time      : début de la fenêtre temporelle (0 si non applicable)
  due_date        : fin de la fenêtre temporelle (inf si non applicable)
  service_time    : temps de service (0 si non applicable)
─────────────────────────────────────────────────────────────────

Usage :
  # Normaliser dataset_raw complet
  python normalize_datasets.py --input ./dataset_raw --output ./dataset_normalized

  # Normaliser un split spécifique
  python normalize_datasets.py --input ./dataset_split/split_70_15_15/train --output ./dataset_normalized/train
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# Schéma de sortie canonique
# ─────────────────────────────────────────────

CANONICAL_COLUMNS = [
    "source",
    "instance_id",
    "category",
    "node_id",
    "x",
    "y",
    "demand",
    "ready_time",
    "due_date",
    "service_time",
]


# ─────────────────────────────────────────────
# Parsers par format
# ─────────────────────────────────────────────

def parse_solomon_csv(filepath: Path) -> pd.DataFrame:
    """
    Parse les fichiers CSV Solomon (solomon_dataset).

    Format attendu :
        CUST NO., XCOORD., YCOORD., DEMAND, READY TIME, DUE DATE, SERVICE TIME
        1,        45,       75,      10,      0,          100,       5
    La ligne 0 est le dépôt (demand=0, time window = planning horizon).
    """
    try:
        df_raw = pd.read_csv(filepath, skipinitialspace=True)
        df_raw.columns = [c.strip().upper() for c in df_raw.columns]

        # Mapping flexible des noms de colonnes
        col_map = {
            "CUST NO.": "node_id",
            "CUST_NO":  "node_id",
            "XCOORD.":  "x",
            "XCOORD":   "x",
            "YCOORD.":  "y",
            "YCOORD":   "y",
            "DEMAND":   "demand",
            "READY TIME": "ready_time",
            "READY_TIME": "ready_time",
            "DUE DATE": "due_date",
            "DUE_DATE":  "due_date",
            "SERVICE TIME": "service_time",
            "SERVICE_TIME": "service_time",
        }
        df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

        needed = ["node_id", "x", "y", "demand", "ready_time", "due_date", "service_time"]
        for col in needed:
            if col not in df_raw.columns:
                df_raw[col] = 0.0

        df = df_raw[needed].copy()
        # Forcer la conversion numérique (protège contre les colonnes lues comme str)
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["source"]      = filepath.name
        df["instance_id"] = filepath.stem
        df["category"]    = filepath.parent.name   # C1, R2, RC1, etc.
        return df[CANONICAL_COLUMNS]

    except Exception as e:
        print(f"  [WARN] Impossible de parser {filepath.name} : {e}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)


def parse_solomon_tsptw_txt(filepath: Path) -> pd.DataFrame:
    """
    Parse les fichiers .txt SolomonTSPTW.

    Format attendu (variable selon instance) :
      - Certains débutent par un entête texte, d'autres directement par les données
      - Colonnes : id  x  y  demand  ready_time  due_date  service_time
    """
    try:
        rows = []
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # On accepte les lignes avec 7 valeurs numériques
                if len(parts) >= 7:
                    try:
                        vals = [float(p) for p in parts[:7]]
                        rows.append(vals)
                    except ValueError:
                        continue   # ligne d'en-tête textuelle, on ignore

        if not rows:
            print(f"  [WARN] Aucune donnée numérique dans {filepath.name}")
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = pd.DataFrame(rows, columns=["node_id", "x", "y", "demand",
                                         "ready_time", "due_date", "service_time"])
        df["source"]      = filepath.name
        df["instance_id"] = filepath.stem
        # Famille : rc_201, rc_202, … (partie avant le dernier point)
        df["category"]    = "tsptw_" + ".".join(filepath.stem.split(".")[:1])
        return df[CANONICAL_COLUMNS]

    except Exception as e:
        print(f"  [WARN] Impossible de parser {filepath.name} : {e}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)


def parse_tsplib(filepath: Path) -> pd.DataFrame:
    """
    Parse les fichiers au format TSPLIB (.tsp, world.tsp, etc.).

    Supporte :
      - NODE_COORD_SECTION  (coordonnées 2D : id x y)
      - EDGE_WEIGHT_SECTION (matrice de distances → on n'extrait pas les coords)
    """
    try:
        nodes = []
        in_coord_section = False
        edge_weight_type = "EUC_2D"

        with open(filepath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                upper = line.upper()

                if upper.startswith("EDGE_WEIGHT_TYPE"):
                    edge_weight_type = line.split(":")[-1].strip().upper()
                    continue
                if upper == "NODE_COORD_SECTION":
                    in_coord_section = True
                    continue
                if upper in ("EOF", "EDGE_WEIGHT_SECTION", "TOUR_SECTION"):
                    in_coord_section = False
                    continue

                if in_coord_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            node_id = int(parts[0])
                            x, y = float(parts[1]), float(parts[2])
                            nodes.append({"node_id": node_id, "x": x, "y": y})
                        except ValueError:
                            continue

        if not nodes:
            print(f"  [WARN] Pas de NODE_COORD_SECTION dans {filepath.name}")
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = pd.DataFrame(nodes)
        df["demand"]       = 0.0
        df["ready_time"]   = 0.0
        df["due_date"]     = float("inf")
        df["service_time"] = 0.0
        df["source"]       = filepath.name
        df["instance_id"]  = filepath.stem
        df["category"]     = "tsplib"
        return df[CANONICAL_COLUMNS]

    except Exception as e:
        print(f"  [WARN] Impossible de parser {filepath.name} : {e}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)


def parse_generic_csv(filepath: Path) -> pd.DataFrame:
    """
    Parser de secours pour les CSV non-standard (ex: tsp_dataset.csv).
    Tente de détecter les colonnes automatiquement.
    """
    try:
        df_raw = pd.read_csv(filepath, skipinitialspace=True)
        df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]

        # Tentative de mapping automatique
        x_candidates = ["x", "xcoord", "longitude", "lon", "lng", "coord_x"]
        y_candidates = ["y", "ycoord", "latitude",  "lat", "coord_y"]

        x_col = next((c for c in x_candidates if c in df_raw.columns), None)
        y_col = next((c for c in y_candidates if c in df_raw.columns), None)

        if x_col is None or y_col is None:
            # Dernier recours : on prend les 2e et 3e colonnes numériques
            num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                x_col, y_col = num_cols[0], num_cols[1]
            else:
                print(f"  [WARN] Colonnes x/y introuvables dans {filepath.name}")
                return pd.DataFrame(columns=CANONICAL_COLUMNS)

        def _col(name, default):
            if name in df_raw.columns:
                return pd.to_numeric(df_raw[name], errors="coerce").fillna(default)
            return pd.Series([default] * len(df_raw), dtype=float)

        df = pd.DataFrame()
        df["node_id"]     = _col("node_id", 0) if "node_id" in df_raw.columns else (
                            _col("id", 0) if "id" in df_raw.columns else
                            pd.RangeIndex(len(df_raw)))
        df["x"]           = pd.to_numeric(df_raw[x_col], errors="coerce").fillna(0.0)
        df["y"]           = pd.to_numeric(df_raw[y_col], errors="coerce").fillna(0.0)
        df["demand"]      = _col("demand",       0.0)
        df["ready_time"]  = _col("ready_time",   0.0)
        df["due_date"]    = _col("due_date",     float("inf"))
        df["service_time"]= _col("service_time", 0.0)
        df["source"]      = filepath.name
        df["instance_id"] = filepath.stem
        df["category"]    = "generic_tsp"
        return df[CANONICAL_COLUMNS]

    except Exception as e:
        print(f"  [WARN] Impossible de parser {filepath.name} : {e}")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)


# ─────────────────────────────────────────────
# Normalisation des valeurs numériques
# ─────────────────────────────────────────────

def normalize_instance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les coordonnées x, y entre 0 et 1 par instance.
    Les fenêtres temporelles sont normalisées par rapport au max due_date fini.
    Les demandes sont normalisées par rapport à la somme totale.
    """
    df = df.copy()

    # --- Coordonnées spatiales ---
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    df["x"] = (df["x"] - x_min) / x_range
    df["y"] = (df["y"] - y_min) / y_range

    # --- Fenêtres temporelles ---
    finite_due = df["due_date"].replace(float("inf"), np.nan).dropna()
    if not finite_due.empty:
        t_max = finite_due.max()
        if t_max > 0:
            df["ready_time"]  = df["ready_time"] / t_max
            df["due_date"]    = df["due_date"].replace(float("inf"), np.nan) / t_max
            df["service_time"]= df["service_time"] / t_max
            # Remplacer les NaN (inf normalisés) par -1 (indicateur "pas de contrainte")
            df["due_date"]    = df["due_date"].fillna(-1.0)

    # --- Demandes ---
    total_demand = df["demand"].sum()
    if total_demand > 0:
        df["demand"] = df["demand"] / total_demand

    return df


# ─────────────────────────────────────────────
# Dispatcher par extension / chemin
# ─────────────────────────────────────────────

def dispatch_parser(filepath: Path) -> pd.DataFrame:
    """Choisit le bon parser selon le chemin et l'extension du fichier."""
    ext  = filepath.suffix.lower()
    name = filepath.name.lower()
    parts = [p.lower() for p in filepath.parts]

    # Fichiers TSPLIB (.tsp)
    if ext == ".tsp" or name.endswith(".tsp"):
        return parse_tsplib(filepath)

    # Fichiers SolomonTSPTW (.txt)
    if ext == ".txt" and "solomontsptw" in parts:
        return parse_solomon_tsptw_txt(filepath)

    # Fichiers CSV Solomon standard
    if ext == ".csv" and "solomon_dataset" in parts:
        return parse_solomon_csv(filepath)

    # CSV générique (tsp_dataset.csv, etc.)
    if ext == ".csv":
        return parse_generic_csv(filepath)

    print(f"  [SKIP] Format non supporté : {filepath.name}")
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


# ─────────────────────────────────────────────
# Conversion de format brut → CSV
# ─────────────────────────────────────────────

def convert_txt_to_csv(filepath: Path) -> bool:
    """
    Convertit un fichier SolomonTSPTW .txt en .csv dans le même dossier.
    Colonnes : node_id, x, y, demand, ready_time, due_date, service_time
    """
    rows = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    vals = [float(p) for p in parts[:7]]
                    rows.append(vals)
                except ValueError:
                    continue
    if not rows:
        return False
    out = filepath.with_suffix(".csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("node_id,x,y,demand,ready_time,due_date,service_time\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")
    return True


def convert_tsp_to_csv(filepath: Path) -> bool:
    """
    Convertit un fichier TSPLIB .tsp en .csv dans le même dossier.
    Colonnes : node_id, x, y
    """
    nodes = []
    in_section = False
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            upper = stripped.upper()
            if upper == "NODE_COORD_SECTION":
                in_section = True
                continue
            if upper in ("EOF", "EDGE_WEIGHT_SECTION", "TOUR_SECTION"):
                in_section = False
                continue
            if in_section:
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        nodes.append((int(parts[0]), float(parts[1]), float(parts[2])))
                    except ValueError:
                        continue
    if not nodes:
        return False
    out = filepath.with_suffix(".csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("node_id,x,y\n")
        for node_id, x, y in nodes:
            fh.write(f"{node_id},{x},{y}\n")
    return True


def convert_raw_formats(input_dir: Path):
    """
    Convertit tous les fichiers .txt et .tsp de input_dir en CSV (même dossier, même nom).
    Les fichiers originaux sont conservés.
    """
    converted, failed, skipped = 0, 0, 0
    for f in sorted(input_dir.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext == ".txt":
            print(f"  → {f.relative_to(input_dir)}", end="  ")
            if convert_txt_to_csv(f):
                print(f"✔  →  {f.stem}.csv")
                converted += 1
            else:
                print("⚠ vide ou non parseable")
                failed += 1
        elif ext == ".tsp":
            print(f"  → {f.relative_to(input_dir)}", end="  ")
            if convert_tsp_to_csv(f):
                print(f"✔  →  {f.stem}.csv")
                converted += 1
            else:
                print("⚠ vide ou non parseable")
                failed += 1
        else:
            skipped += 1

    print(f"\n  ✔ Convertis : {converted}  ⚠ Échecs : {failed}  ─ Ignorés : {skipped}")


# ─────────────────────────────────────────────
# Moteur principal
# ─────────────────────────────────────────────

def collect_all_files(input_dir: Path) -> list:
    """Collecte récursivement tous les fichiers supportés."""
    supported = {".csv", ".txt", ".tsp"}
    files = []
    for f in sorted(input_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in supported:
            # Ignorer les fichiers déjà normalisés
            if "normalized" in f.name.lower():
                continue
            files.append(f)
    return files


def normalize_all(input_dir: Path, output_dir: Path, per_instance: bool = True):
    """
    Parse, normalise et exporte tous les fichiers trouvés dans input_dir.

    Génère :
      - Un CSV par instance dans output_dir/instances/
      - Un CSV global consolidé : output_dir/all_instances.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(exist_ok=True)

    files = collect_all_files(input_dir)
    print(f"\n  Fichiers détectés : {len(files)}")

    all_frames = []
    stats = {"ok": 0, "empty": 0, "error": 0}

    for filepath in files:
        print(f"  → {filepath.relative_to(input_dir)}", end="  ")

        df = dispatch_parser(filepath)

        if df.empty:
            print("⚠ vide")
            stats["empty"] += 1
            continue

        # Normalisation numérique par instance
        if per_instance:
            df = normalize_instance(df)

        # Sauvegarde CSV individuel
        out_name = filepath.stem + "_normalized.csv"
        out_path = instances_dir / out_name
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"✔ ({len(df)} nœuds)")

        all_frames.append(df)
        stats["ok"] += 1

    # CSV global consolidé
    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
        global_path = output_dir / "all_instances.csv"
        df_all.to_csv(global_path, index=False, float_format="%.6f")
        print(f"\n  ✔ CSV global : {global_path}  ({len(df_all)} lignes, {df_all['instance_id'].nunique()} instances)")
    else:
        print("\n  [ERROR] Aucune donnée valide trouvée.")

    # Résumé
    print(f"\n  {'─'*40}")
    print(f"  Résumé de la normalisation")
    print(f"  {'─'*40}")
    print(f"  ✔ Succès   : {stats['ok']}")
    print(f"  ⚠ Vides    : {stats['empty']}")
    print(f"  ✘ Erreurs  : {stats['error']}")
    if all_frames:
        print(f"\n  Colonnes  : {CANONICAL_COLUMNS}")
        print(f"  Catégories: {sorted(df_all['category'].unique())}")
        print(f"  Instances : {df_all['instance_id'].nunique()}")
        print(f"  Nœuds tot.: {len(df_all)}")

    return df_all if all_frames else pd.DataFrame(columns=CANONICAL_COLUMNS)


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Normalise tous les datasets TSP/VRPTW vers un format CSV unifié."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./dataset_raw",
        help="Dossier source (défaut : ./dataset_raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset_normalized",
        help="Dossier de sortie (défaut : ./dataset_normalized)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Désactiver la normalisation min-max (garder les valeurs brutes)",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convertir les fichiers .txt et .tsp en CSV dans le dossier source (avant normalisation)",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    per_instance = not args.no_normalize

    print(f"\n  Source  : {input_dir}")
    print(f"  Sortie  : {output_dir}")
    print(f"  Normalisation min-max : {'OUI' if per_instance else 'NON'}")

    if not input_dir.exists():
        print(f"\n  [ERROR] Le dossier source n'existe pas : {input_dir}")
        return

    if args.convert:
        print(f"\n{'─'*55}")
        print("  Conversion .txt / .tsp → CSV")
        print(f"{'─'*55}")
        convert_raw_formats(input_dir)
        print(f"\n  ✅ Conversion terminée !\n")

    normalize_all(input_dir, output_dir, per_instance=per_instance)
    print(f"\n  ✅ Normalisation terminée !\n")


if __name__ == "__main__":
    main()