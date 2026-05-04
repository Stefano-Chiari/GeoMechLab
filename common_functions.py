import numpy as np

import pygeon as pg

from elastic_pb import ElasticProblem


"""Shared utilities for elasticity solver entry points.

These helpers keep all main scripts consistent in boundary conditions,
material setup, solve workflow, and export behavior.
"""

# Definisce una forza costante applicata al dominio, usata come condizione al contorno di Neumann
def _default_traction(_):
    """Return the default constant traction vector.

    Parameters
    ----------
    _ : np.ndarray
        Unused input accepted for compatibility with callback signatures.

    Returns
    -------
    np.ndarray
        Constant traction vector ``[fx, fy]``.
    """
    return np.array([0.0, -1e-3])       # x = 0.0, y = -1e-3


# Identifica geometricamente quali parti del dominio si trovano sul bordo inferiore (block) e sul bordo superiore (apply the force)
def _standard_boundary_masks(sd):
    """Build standard bottom-Dirichlet and top-Neumann masks.

    Parameters
    ----------
    sd : pg.Grid
        Grid object containing node coordinates and face centers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bottom, top)`` boolean masks, where ``bottom`` selects displacement
        dofs constrained by essential conditions and ``top`` selects faces with
        natural traction conditions.
    """
    # Use geometric min/max so this also works for non-unit domains.
    y_min = float(np.min(sd.nodes[1, :]))                   # coordinata y minima tra tutti nodi griglia
    y_max = float(np.max(sd.face_centers[1, :]))            # coordinata y massima tra tutti nodi griglia

    # Essential BC acts on vector dofs, so node mask is repeated by sd.dim.
    bottom = np.hstack([np.isclose(sd.nodes[1, :], y_min)] * sd.dim)        # maschera booleana per i nodi inferiori
    top = np.isclose(sd.face_centers[1, :], y_max)                          # maschera booleana per facce superiori a y_max

    return bottom, top

# Prende griglia, imposta i materiali, risolve equazioni elasticità e salva i risultati
def solve_and_export(sd, folder_export, export_name, cell_data=None):
    """Solve linear elasticity on a grid and export solution fields.

    Parameters
    ----------
    sd : pg.Grid
        Grid used for discretization and assembly.
    folder_export : str or pathlib.Path
        Output directory where exported files are written.
    export_name : str
        Base name used by the exporter for output files.
    cell_data : list[tuple[str, np.ndarray]] or None, optional
        Optional additional cell fields to export.

    Returns
    -------
    np.ndarray
        Displacement field returned by the linear solver.
    """
    # Step 1: define material parameters and standard boundary masks.
    lambda_ = 1.0       # lambda per il materiale elastico
    mu = 0.5            # modulo di taglio
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}   # dizionario con parametri materiali
    bottom, top = _standard_boundary_masks(sd)          # maschere dei bordi (da function precedente)

    # Step 2: assemble and solve the discrete elasticity system.
    elastic_pb = ElasticProblem(sd)                                     # inizializza oggetto passandogli la griglia
    A, b = elastic_pb.assemble_problem(param, _default_traction, top)   # matrice di rigidezza e vettore carichi, usando facce superiori e trazione
    u, sigma = elastic_pb.solve_linear_system(A, b, bottom)             # risolve sis. lin. con cond. Dirichlet. Restituisce spostamenti e sforzi

    # Step 3: export through ElasticProblem only (single exporter path).
    elastic_pb.export_solution(
        u,
        sigma,
        folder_export,
        export_name=export_name,
        cell_data=cell_data,
    )
    # restituisce campo degli spostamenti calcolato
    return u

def solve_and_export_pressure(sd, layers, layer_ID, pressure_val, folder_export, export_name, cell_data=None):
    """Solve linear elasticity on a grid and export solution fields.

    Parameters
    ----------
    sd : pg.Grid
        Grid used for discretization and assembly.
    folder_export : str or pathlib.Path
        Output directory where exported files are written.
    export_name : str
        Base name used by the exporter for output files.
    cell_data : list[tuple[str, np.ndarray]] or None, optional
        Optional additional cell fields to export.

    Returns
    -------
    np.ndarray
        Displacement field returned by the linear solver.
    """
    # Step 1: define material parameters and standard boundary masks.
    lambda_ = 1.0
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}
    bottom, top = _standard_boundary_masks(sd)
    sd.compute_geometry()

    pressure = np.array(pressure_val*(layers==layer_ID))

    # Step 2: assemble and solve the discrete elasticity system.
    elastic_pb = ElasticProblem(sd)
    A, b = elastic_pb.assemble_problem_pressure(param, _default_traction,pressure, top)
    u, sigma = elastic_pb.solve_linear_system(A, b, bottom)

    # Step 3: export through ElasticProblem only (single exporter path).
    elastic_pb.export_solution(
        u,
        sigma,
        folder_export,
        export_name=export_name,
        cell_data=cell_data,
    )

    return u

