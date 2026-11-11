"""The cell's final state as arrays, and the two ParaView writers.

Ported from the parent's ``scripts/run/run_micro_cells.py`` at 2aaccad: ``_extract_fields``
(143), ``_b64``, ``_write_vts`` (242), ``_write_xdmf`` (267).  The bodies are the parent's;
the only change is that ``extract_fields`` takes ``mixed`` (a bool) where the parent took
the ``MicroCase``.  ``to_si`` is new: the display units (MPa, nm) back to SI, with the
arithmetic ``tools/make_references.py`` used for the reference files.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

NM = 1.0e9
MPA = 1.0e-6


def extract_fields(s, mixed: bool) -> tuple[dict, dict, dict]:
    """The solver's final state on the (n+1) x (n+1) vertex grid ([i_x, j_y], display units:
    MPa and nm), the DG0 fluid indicator, and consistency diagnostics."""
    from dolfin import Constant, project

    n, _i, _j = s._periodic_grid()
    wrap = np.arange(n + 1) % n
    V = s.Vfull
    X = V.tabulate_dof_coordinates().reshape(-1, 2)
    I = np.rint(X[:, 0] * n).astype(int)
    J = np.rint(X[:, 1] * n).astype(int)

    def from_full(v):
        G = np.full((n + 1, n + 1), np.nan)
        G[I, J] = v
        return G

    def from_per(dofs):
        return s._dofs_to_grid(np.asarray(dofs, dtype=float))[np.ix_(wrap, wrap)]

    def from_grid(g):
        return np.asarray(g, dtype=float)[np.ix_(wrap, wrap)]

    p = s.params
    H0, HT, T = float(p.H0), float(p.HT), float(s.T_dim_const)
    h_min = float(p.h_min)
    out: dict = {}
    info: dict = {"n": int(n), "T_end_s": T}

    P = from_full(project(s.P_dim, V).vector().get_local())
    film_proj = from_full(project(Constant(H0) * s.h_nd, V).vector().get_local())
    rigid = from_full(project(Constant(H0) * s.h_base_nd, V).vector().get_local())

    ripple_on = bool(getattr(s, "_ripple_on", False)) and getattr(s, "ur_nd_func", None) is not None
    u_r = from_per(s.ur_nd_func.vector().get_local()) * H0 if ripple_on else None

    fft = getattr(s, "_contact_solution", None) is not None and getattr(s, "uc_nd_func", None) is not None
    if fft:
        sol = s._contact_solution
        rigid = from_grid(s._rigid_film_grid())          # the contact's own rigid film (exact, nodal)
        u_c = from_grid(sol.u)
        p_c = from_grid(sol.p)
        phi_c = from_grid(sol.phi_c)
        active = from_grid(np.asarray(sol.active, dtype=float))
        film = rigid + (u_r if u_r is not None else 0.0) + u_c
        info["film_vs_solver_max_nm"] = float(np.nanmax(np.abs(film - film_proj)) * NM)
        info["contact_residual_m"] = float(sol.residual)
        chi = (phi_c > 0.0).astype(float)
        out["p_contact_MPa"] = p_c * MPA
        out["clearance_nm"] = (film - h_min) * NM
        out["mask_level_nm"] = phi_c * NM
        out["overclosure_nm"] = np.maximum(h_min - (rigid + (u_r if u_r is not None else 0.0)), 0.0) * NM
        contact = active
    else:
        film = film_proj
        if bool(mixed) and h_min > 0.0:
            chi = (film - h_min > 0.0).astype(float)
            out["clearance_nm"] = (film - h_min) * NM
        else:
            chi = np.ones_like(film)
        contact = 1.0 - chi
        p_c = None

    out["p_fluid_MPa"] = P * MPA
    out["p_fluid_masked_MPa"] = np.where(chi > 0.5, P, np.nan) * MPA
    out["p_total_MPa"] = (np.where(chi > 0.5, P, 0.0) + (p_c if p_c is not None else 0.0)) * MPA
    out["film_nm"] = film * NM
    out["film_rigid_nm"] = rigid * NM
    out["surface_height_nm"] = -(rigid - (H0 + HT * T)) * NM
    if u_r is not None:
        out["u_ripple_nm"] = u_r * NM
    if fft:
        out["u_contact_nm"] = u_c * NM
        out["u_total_nm"] = ((u_r if u_r is not None else 0.0) + u_c) * NM
    out["contact"] = contact
    out["chi_fluid"] = chi

    chi_dg0 = project(s.chi_fluid, s.Vdg0).vector().get_local()
    info.update({
        "film_min_nm": float(np.nanmin(film) * NM),
        "film_rigid_min_nm": float(np.nanmin(rigid) * NM),
        "p_fluid_max_MPa": float(np.nanmax(P) * MPA),
        "p_contact_max_MPa": float(np.nanmax(p_c) * MPA) if p_c is not None else 0.0,
        "contact_nodes": int(np.sum(contact[:n, :n] > 0.5)),
        "u_ripple_maxabs_nm": float(np.nanmax(np.abs(u_r)) * NM) if u_r is not None else 0.0,
        "u_contact_maxabs_nm": float(np.nanmax(np.abs(u_c)) * NM) if fft else 0.0,
        "chi_dg0_mean": float(np.mean(chi_dg0)),
    })
    for k, G in out.items():
        if np.isnan(G).all():
            raise RuntimeError(f"field {k} is all NaN")
    return out, {"chi_fluid_dg0": chi_dg0}, info


def to_si(fields: dict) -> dict:
    """The display-unit fields of :func:`extract_fields` in SI, the unit suffix dropped
    (``p_fluid_MPa`` -> ``p_fluid`` [Pa], ``film_nm`` -> ``film`` [m]); the dimensionless
    ``contact`` and ``chi_fluid`` unchanged."""
    out = {}
    for name, G in fields.items():
        if name.endswith("_MPa"):
            out[name[:-4]] = G / MPA
        elif name.endswith("_nm"):
            out[name[:-3]] = G / NM
        else:
            out[name] = G
    return out


def _b64(a, dtype="<f8") -> str:
    raw = np.ascontiguousarray(a, dtype=dtype).tobytes()
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode("ascii")


def write_vts(path: Path, fields: dict, Lx_um: float, Ly_um: float) -> None:
    """VTK XML structured grid, binary inline (NaN-safe), x fastest."""
    n1 = next(iter(fields.values())).shape[0]
    n = n1 - 1
    XX, YY = np.meshgrid(np.linspace(0.0, Lx_um, n1), np.linspace(0.0, Ly_um, n1), indexing="ij")
    pts = np.zeros((n1 * n1, 3))
    pts[:, 0] = XX.ravel(order="F")
    pts[:, 1] = YY.ravel(order="F")
    ext = f"0 {n} 0 {n} 0 0"
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">',
             f'  <StructuredGrid WholeExtent="{ext}">',
             f'    <Piece Extent="{ext}">',
             f'      <PointData Scalars="{next(iter(fields))}">']
    for name, G in fields.items():
        lines.append(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="binary">')
        lines.append("          " + _b64(G.ravel(order="F")))
        lines.append("        </DataArray>")
    lines += ["      </PointData>", "      <Points>",
              '        <DataArray type="Float64" NumberOfComponents="3" format="binary">',
              "          " + _b64(pts.ravel()),
              "        </DataArray>", "      </Points>", "    </Piece>", "  </StructuredGrid>", "</VTKFile>", ""]
    Path(path).write_text("\n".join(lines))


def write_xdmf(path: Path, s, fields: dict, cell_fields: dict, Lx_um: float, Ly_um: float) -> None:
    """The fields as CG1 functions (and DG0 cell data) on a copy of the micro mesh
    scaled to um, one XDMF file per cell (functions share the mesh)."""
    from dolfin import MPI, Function, FunctionSpace, Mesh, XDMFFile

    n = next(iter(fields.values())).shape[0] - 1
    mesh = Mesh(s.mesh_m)
    xy = mesh.coordinates()
    xy[:, 0] *= Lx_um
    xy[:, 1] *= Ly_um
    V = FunctionSpace(mesh, "CG", 1)
    X = V.tabulate_dof_coordinates().reshape(-1, 2)
    I = np.rint(X[:, 0] / Lx_um * n).astype(int)
    J = np.rint(X[:, 1] / Ly_um * n).astype(int)
    # DG0: match cells by midpoint (triangle midpoints sit on the 1/(3n) lattice)
    V0 = FunctionSpace(mesh, "DG", 0)
    X0 = V0.tabulate_dof_coordinates().reshape(-1, 2)
    X0s = s.Vdg0.tabulate_dof_coordinates().reshape(-1, 2)
    key_new = {(int(round(a / Lx_um * 3 * n)), int(round(b / Ly_um * 3 * n))): k for k, (a, b) in enumerate(X0)}
    perm = np.array([key_new[(int(round(a * 3 * n)), int(round(b * 3 * n)))] for a, b in X0s])
    with XDMFFile(MPI.comm_self, str(path)) as xf:
        xf.parameters["functions_share_mesh"] = True
        xf.parameters["rewrite_function_mesh"] = False
        xf.parameters["flush_output"] = True
        for name, G in fields.items():
            f = Function(V)
            f.rename(name, name)
            f.vector().set_local(G[I, J])
            f.vector().apply("insert")
            xf.write(f, 0.0)
        for name, vals in cell_fields.items():
            f = Function(V0)
            f.rename(name, name)
            arr = np.empty(V0.dim())
            arr[perm] = vals
            f.vector().set_local(arr)
            f.vector().apply("insert")
            xf.write(f, 0.0)
