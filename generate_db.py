import pandas as pd
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from inverse_cavity import _run_forward, MESH_FINE, LX, LY, MARGIN_X, MARGIN_Y
import multiprocessing as mp

def build_dense_grid(nx: int = 25, ny: int = 15) -> list[tuple[float, float]]:
    xs = np.linspace(MARGIN_X, LX - MARGIN_X, nx)
    ys = np.linspace(MARGIN_Y, LY - MARGIN_Y, ny)
    return [(float(x), float(y)) for x in xs for y in ys]

def worker_db(args):
    cx, cy, mesh_size = args
    import os, tempfile
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            ux, uy = _run_forward((cx, cy), mesh_size)
        finally:
            os.chdir(original_cwd)
    return cx, cy, ux, uy

def generate_database(filename: str = "cavity_db.csv", nx: int = 25, ny: int = 15):
    candidates = build_dense_grid(nx, ny)
    print(f"Avvio generazione database per {len(candidates)} configurazioni...")
    t0 = time.perf_counter()
    
    args_list = [(cx, cy, MESH_FINE) for (cx, cy) in candidates]
    results = []
    
    # Esecuzione parallela
    with ProcessPoolExecutor() as pool:
        future_map = {pool.submit(worker_db, args): args for args in args_list}
        for future in as_completed(future_map):
            try:
                cx, cy, ux, uy = future.result()
                if not np.isnan(ux):
                    results.append({"center_x": cx, "center_y": cy, "ux": ux, "uy": uy})
            except Exception as e:
                pass
                
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    
    t1 = time.perf_counter()
    print(f"Database salvato in '{filename}' con {len(df)} entry valide.")
    print(f"Tempo di esecuzione: {t1 - t0:.1f} secondi.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    generate_database()