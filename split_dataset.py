"""
Dataset Splitter for TSP/VRPTW instances
=========================================
Crée deux splits équilibrés (stratifiés) à partir de dataset_raw :
  - Split A : 60% train / 20% val / 20% test
  - Split B : 70% train / 15% val / 15% test

Structure attendue de dataset_raw :
  dataset_raw/
  ├── SolomonTSPTW/SolomonTSPTW/*.txt     (groupé par famille rc_2XX)
  ├── solomon_dataset/{C1,C2,R1,R2,RC1,RC2}/*.csv
  └── chunks/
      ├── tsp_dataset/tsp_dataset.csv     (fichier unique fusionné)
      └── world/world.csv                 (fichier unique fusionné)

Usage :
  python split_dataset.py [--raw_dir ./dataset_raw] [--out_dir ./dataset_split] [--chunk_size 20000]
"""

import os
import shutil
import random
import argparse
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SPLITS = {
    "split_60_20_20": (0.60, 0.20, 0.20),
    "split_70_15_15": (0.70, 0.15, 0.15),
}

RANDOM_SEED = 42


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def count_data_rows(filepath: str) -> int:
    """
    Compte les lignes de données dans un fichier (hors en-tête).
    Supporte .csv, .txt (SolomonTSPTW) et .tsp (TSPLIB).
    """
    p = Path(filepath)
    ext = p.suffix.lower()
    try:
        if ext == ".csv":
            with open(p, encoding="utf-8", errors="replace") as fh:
                lines = [l for l in fh if l.strip()]
            return max(0, len(lines) - 1)   # -1 pour l'en-tête

        if ext == ".txt":
            count = 0
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        try:
                            [float(x) for x in parts[:7]]
                            count += 1
                        except ValueError:
                            pass
            return count

        if ext == ".tsp":
            count = 0
            in_section = False
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    upper = line.strip().upper()
                    if upper == "NODE_COORD_SECTION":
                        in_section = True
                        continue
                    if upper in ("EOF", "EDGE_WEIGHT_SECTION", "TOUR_SECTION"):
                        in_section = False
                        continue
                    if in_section:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            try:
                                int(parts[0]); float(parts[1]); float(parts[2])
                                count += 1
                            except ValueError:
                                pass
            return count
    except Exception:
        pass
    return 1


def stratified_split(files: list, train_r: float, val_r: float, test_r: float, seed: int = RANDOM_SEED):
    """
    Découpe une liste de fichiers en train/val/test de façon reproductible.
    Le split est pondéré par le nombre de lignes de données de chaque fichier,
    afin que les proportions reflètent le volume réel et non juste le nombre de fichiers.
    """
    assert abs(train_r + val_r + test_r - 1.0) < 1e-9, "Les ratios doivent sommer à 1.0"
    rng = random.Random(seed)
    files = sorted(files)
    rng.shuffle(files)

    n = len(files)
    if n == 1:
        return files, [], []
    if n == 2:
        return files[:1], files[1:], []
    if n == 3:
        return files[:1], files[1:2], files[2:]

    # Compter les lignes de chaque fichier
    row_counts = [(f, count_data_rows(f)) for f in files]
    total_rows = sum(r for _, r in row_counts)

    target_train = total_rows * train_r
    target_val   = total_rows * val_r

    train, val, test = [], [], []
    cum_train, cum_val = 0, 0

    for f, rows in row_counts:
        if cum_train < target_train:
            train.append(f)
            cum_train += rows
        elif cum_val < target_val:
            val.append(f)
            cum_val += rows
        else:
            test.append(f)

    # Garantir qu'aucun split n'est vide
    if not val and len(train) > 1:
        val.append(train.pop())
    if not test and len(train) > 1:
        test.append(train.pop())

    return train, val, test


def copy_files(file_list: list, src_base: Path, dst_dir: Path, preserve_subpath: bool = True):
    """
    Copie les fichiers dans dst_dir en conservant (optionnellement) leur sous-chemin relatif.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in file_list:
        f = Path(f)
        if preserve_subpath:
            rel = f.relative_to(src_base)
            dest = dst_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest = dst_dir / f.name
        shutil.copy2(f, dest)


def print_summary(label: str, groups: dict):
    """Affiche un résumé du split par groupe (fichiers et lignes de données)."""
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    # En-tête
    print(f"  {'Groupe':<25} {'Train':>7} {'Val':>7} {'Test':>7}  {'Total':>7}  {'Lignes':>10}")
    print(f"  {'─'*25} {'─'*7} {'─'*7} {'─'*7}  {'─'*7}  {'─'*10}")
    totals_files = [0, 0, 0]
    totals_rows  = [0, 0, 0]
    for grp, (tr, va, te) in sorted(groups.items()):
        r_tr = sum(count_data_rows(f) for f in tr)
        r_va = sum(count_data_rows(f) for f in va)
        r_te = sum(count_data_rows(f) for f in te)
        total_r = r_tr + r_va + r_te
        print(
            f"  {grp:<25} {len(tr):>7} {len(va):>7} {len(te):>7}  "
            f"{len(tr)+len(va)+len(te):>7}  {total_r:>10,}"
        )
        totals_files[0] += len(tr);   totals_files[1] += len(va);   totals_files[2] += len(te)
        totals_rows[0]  += r_tr;      totals_rows[1]  += r_va;      totals_rows[2]  += r_te
    tf = sum(totals_files); tr_total = sum(totals_rows)
    print(f"  {'TOTAL fichiers':<25} {totals_files[0]:>7} {totals_files[1]:>7} {totals_files[2]:>7}  {tf:>7}  {tr_total:>10,}")
    pct_f = [f"{100*t/tf:.1f}%" if tf else "─" for t in totals_files]
    pct_r = [f"{100*t/tr_total:.1f}%" if tr_total else "─" for t in totals_rows]
    print(f"  {'  % fichiers':<25} {pct_f[0]:>7} {pct_f[1]:>7} {pct_f[2]:>7}")
    print(f"  {'  % lignes':<25} {pct_r[0]:>7} {pct_r[1]:>7} {pct_r[2]:>7}")


# ─────────────────────────────────────────────
# Collecteurs de fichiers par source
# ─────────────────────────────────────────────

def collect_solomon_dataset(raw_dir: Path) -> dict:
    """
    solomon_dataset/{C1,C2,R1,R2,RC1,RC2}/*.csv
    → une entrée par catégorie (C1, R2, etc.)
    """
    base = raw_dir / "solomon_dataset"
    groups = {}
    if not base.exists():
        print(f"  [WARN] Dossier introuvable : {base}")
        return groups
    for cat_dir in sorted(base.iterdir()):
        if cat_dir.is_dir():
            files = sorted(cat_dir.glob("*.csv"))
            if files:
                groups[f"solomon_{cat_dir.name}"] = [str(f) for f in files]
    return groups


def collect_solomon_tsptw(raw_dir: Path) -> dict:
    """
    SolomonTSPTW/SolomonTSPTW/*.csv  (ou *.txt si pas encore convertis)
    → une entrée par famille (rc_201, rc_202, …)
    """
    base = raw_dir / "SolomonTSPTW" / "SolomonTSPTW"
    groups = defaultdict(list)
    if not base.exists():
        print(f"  [WARN] Dossier introuvable : {base}")
        return {}
    # Préférer .csv (converti) ; fallback sur .txt (brut)
    seen_stems: set = set()
    for ext in ("*.csv", "*.txt"):
        for f in sorted(base.glob(ext)):
            if f.stem not in seen_stems:
                seen_stems.add(f.stem)
                family = ".".join(f.stem.split(".")[:1])   # rc_201
                groups[f"tsptw_{family}"].append(str(f))
    return dict(groups)


def chunk_file_by_rows(filepath: Path, chunk_size: int, out_dir: Path, prefix: str) -> list:
    """
    Découpe un CSV en fichiers de chunk_size lignes (hors en-tête).
    Retourne la liste des fichiers créés.
    Si les chunks existent déjà, les réutilise sans re-créer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        idx = 0
        buf = []
        for line in fh:
            if not line.strip():
                continue
            buf.append(line)
            if len(buf) >= chunk_size:
                chunk_path = out_dir / f"{prefix}_{idx:06d}.csv"
                if not chunk_path.exists():
                    with open(chunk_path, "w", encoding="utf-8") as cf:
                        cf.write(header)
                        cf.writelines(buf)
                chunks.append(str(chunk_path))
                buf = []
                idx += 1
        if buf:
            chunk_path = out_dir / f"{prefix}_{idx:06d}.csv"
            if not chunk_path.exists():
                with open(chunk_path, "w", encoding="utf-8") as cf:
                    cf.write(header)
                    cf.writelines(buf)
            chunks.append(str(chunk_path))
    return chunks


def collect_misc(raw_dir: Path, chunk_size: int) -> dict:
    """
    Découpe les fichiers uniques fusionnés en chunks de chunk_size lignes :
      chunks/tsp_dataset/tsp_dataset.csv  → chunks tsp_XXXXXX.csv
      chunks/world/world.csv              → chunks world_XXXXXX.csv
    Les chunks existants sont supprimés et recréés à chaque appel
    pour garantir la cohérence avec le chunk_size demandé.
    """
    groups = {}
    chunks_dir = raw_dir / "chunks"

    # ── tsp_dataset ───────────────────────────────────────────────
    tsp_src = chunks_dir / "tsp_dataset" / "tsp_dataset.csv"
    tsp_chunks_dir = chunks_dir / "tsp_dataset"
    if tsp_src.exists():
        for old in tsp_chunks_dir.glob("tsp_[0-9]*.csv"):
            old.unlink()
        print(f"  [CHUNK] tsp_dataset.csv → {chunk_size} instances/fichier…", end=" ", flush=True)
        files = chunk_file_by_rows(tsp_src, chunk_size, tsp_chunks_dir, "tsp")
        print(f"{len(files)} fichiers")
        if files:
            groups["misc_tsp"] = files
    else:
        print(f"  [WARN] Fichier introuvable : {tsp_src}")

    # ── world ─────────────────────────────────────────────────────
    world_src = chunks_dir / "world" / "world.csv"
    world_chunks_dir = chunks_dir / "world"
    if world_src.exists():
        for old in world_chunks_dir.glob("world_[0-9]*.csv"):
            old.unlink()
        print(f"  [CHUNK] world.csv → {chunk_size} nœuds/fichier…", end=" ", flush=True)
        files = chunk_file_by_rows(world_src, chunk_size, world_chunks_dir, "world")
        print(f"{len(files)} fichiers")
        if files:
            groups["misc_world"] = files
    else:
        print(f"  [WARN] Fichier introuvable : {world_src}")

    return groups


# ─────────────────────────────────────────────
# Moteur principal
# ─────────────────────────────────────────────

def build_split(
    raw_dir: Path,
    out_dir: Path,
    split_name: str,
    train_r: float,
    val_r: float,
    test_r: float,
    chunk_size: int,
):
    """
    Construit un split complet et copie les fichiers dans out_dir/split_name/.
    """
    print(f"\n{'═'*55}")
    print(f"  Génération de : {split_name}  ({train_r:.0%}/{val_r:.0%}/{test_r:.0%})")
    print(f"{'═'*55}")

    # 1. Collecter tous les groupes
    all_groups: dict = {}
    all_groups.update(collect_solomon_dataset(raw_dir))
    all_groups.update(collect_solomon_tsptw(raw_dir))
    all_groups.update(collect_misc(raw_dir, chunk_size))

    if not all_groups:
        print("  [ERROR] Aucun fichier trouvé. Vérifiez --raw_dir.")
        return

    # 2. Splitter chaque groupe de façon stratifiée
    split_root = out_dir / split_name
    summary = {}

    for grp_name, files in all_groups.items():
        train, val, test = stratified_split(files, train_r, val_r, test_r)
        summary[grp_name] = (train, val, test)

        # 3. Copier les fichiers
        for subset_name, subset_files in [("train", train), ("val", val), ("test", test)]:
            dst = split_root / subset_name
            copy_files(subset_files, raw_dir, dst, preserve_subpath=True)

    # 4. Afficher le résumé
    print_summary(split_name, summary)

    # 5. Écrire un manifeste JSON pour traçabilité
    import json
    manifest = {}
    for grp, (tr, va, te) in summary.items():
        manifest[grp] = {
            "train": [Path(f).name for f in tr],
            "val":   [Path(f).name for f in va],
            "test":  [Path(f).name for f in te],
        }
    manifest_path = split_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\n  ✔ Manifeste écrit : {manifest_path}")
    print(f"  ✔ Fichiers copiés dans : {split_root}")


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crée des splits train/val/test équilibrés pour les datasets TSP/VRPTW."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./dataset_raw",
        help="Chemin vers le dossier dataset_raw (défaut : ./dataset_raw)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./dataset_split",
        help="Dossier de sortie (défaut : ./dataset_split)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20_000,
        help="Nombre max de points (lignes) par fichier chunk (défaut : 20000)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"\n  Source     : {raw_dir}")
    print(f"  Sortie     : {out_dir}")
    print(f"  Chunk size : {args.chunk_size:,} points/fichier")

    if not raw_dir.exists():
        print(f"\n[ERROR] Le dossier source n'existe pas : {raw_dir}")
        return

    # Générer les deux splits
    for split_name, (tr, va, te) in SPLITS.items():
        build_split(raw_dir, out_dir, split_name, tr, va, te, args.chunk_size)

    print(f"\n{'═'*55}")
    print("  ✅  Tous les splits ont été générés avec succès !")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()