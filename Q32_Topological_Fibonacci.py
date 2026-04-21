#!/usr/bin/env python3

"""
Q32 Topological Quantum Computing — Fibonacci anyon braiding
(čisto kvantno: braid-word od Fibonacci R i F matrica + controlled-braid
entanglement između anyonskih qubit-ova, BEZ klasičnog ML-a, BEZ hibrida).

Koncept:
  Fibonacci anyoni su ne-Abelovi anyoni sa fuzionim pravilom:
        τ x τ = 1 ⊕ τ           (NE-trivijalna fuzija → superpozicija kanala!)
        τ x 1 = 1 x τ = τ
  Za n τ-anyona sa totalnim fuzionim kanalom τ ili 1, dimenzija fuzionog Hilbert-
  prostora je Fibonacci broj F_{n-1} (otuda ime). Braid operacije (upletanja anyonskih
  world-lines) generišu UNIVERZALNU kvantnu računsku klasu — bilo koji unitarni
  operator može se aproksimirati braid-wordovima.

  Jezgrene Fibonacci konstante (golden ratio φ = (1+√5)/2):
    R_1   = e^(-4π i / 5)             ← R-matrica u fuzionom kanalu a = 1
    R_τ   = e^( 3π i / 5)             ← R-matrica u fuzionom kanalu a = τ
    F_mat = [[1/φ,         1/√φ     ],
             [1/√φ,        -1/φ     ]]   (F² = I, F je self-inverz)

  Braid generatori na jednom "Fibonacci qubit-u" (3 anyona, 2-dim. fuzioni prostor):
    σ_1 = diag(R_1, R_τ)              ← braid prva 2 anyona (dijagonalan)
    σ_2 = F · σ_1 · F                 ← braid poslednja 2 anyona (kroz F-change)
    [σ_1, σ_2] generiraju gustu podgrupu SU(2) (univerzalnost na jednom qubit-u).

Primena na loto predikciju:
  1) NQ_FIB Fibonacci qubit-a; inicijalno stanje = SP(amp_from_freq(freq_csv))
     — amplitude iz CELOG CSV-a posejane u anyonski prostor.
  2) D_BRAID slojeva braid-wordova. Za svaki sloj ℓ i qubit i:
       Primeni σ_1^(k1[i, ℓ]) zatim σ_2^(k2[i, ℓ]) — single-qubit Fibonacci braid.
     Eksponenti k1, k2 deterministički izvedeni iz CSV freq-segmenata.
  3) Inter-qubit entanglement: za svaki par (i, i+1) controlled-σ_1^(k3[i, ℓ])
     sa ctrl=i, target=i+1 (topologija-inspiracija "long-range braid").
  4) Merenje → amplituda u computational bazi → bias_39 → TOP-7 = NEXT.

Periodičnost braid-eksponenata:
  R_τ^10 = e^(30π i / 5) = e^(6π i) = 1, slično R_1^5 = 1. σ_1^10 = I. 
  Stoga se svi eksponenti uzimaju modulo 10, što pokriva pun ciklus Fibonacci faza.

Qubit budget: NQ_FIB (bez ancilla, bez phase-registra). NQ_FIB = 6 → 64 stanja ≥ 39.

Sve deterministički: seed=39; braid eksponenti izvedeni iz CELOG CSV-a.
Deterministička grid-optimizacija (D_BRAID, braid_scale) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""



from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

NQ_FIB = 6
GRID_D_BRAID = (1, 2, 3, 4)
GRID_BRAID_SCALE = (1, 2)
BRAID_PERIOD = 10

# =========================
# Fibonacci anyon konstante
# =========================
PHI = float((1.0 + np.sqrt(5.0)) / 2.0)
R_1 = complex(np.exp(-4j * np.pi / 5.0))
R_TAU = complex(np.exp(3j * np.pi / 5.0))

F_MAT = np.array(
    [
        [1.0 / PHI, 1.0 / np.sqrt(PHI)],
        [1.0 / np.sqrt(PHI), -1.0 / PHI],
    ],
    dtype=np.complex128,
)


def sigma1_pow(k: int) -> np.ndarray:
    k_mod = int(k) % BRAID_PERIOD
    return np.diag([R_1 ** k_mod, R_TAU ** k_mod]).astype(np.complex128)


def sigma2_pow(k: int) -> np.ndarray:
    return F_MAT @ sigma1_pow(k) @ F_MAT


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Braid eksponenti iz CSV (deterministički)
# =========================
def braid_exponents(H: np.ndarray, nq: int, d_layers: int, scale: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = freq_vector(H)
    edges = np.linspace(0, N_MAX, nq + 1, dtype=int)

    seg_sum = np.array(
        [float(freq[edges[i] : edges[i + 1]].sum()) if edges[i + 1] > edges[i] else 0.0 for i in range(nq)],
        dtype=np.float64,
    )
    seg_mean = np.array(
        [float(freq[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(nq)],
        dtype=np.float64,
    )

    k1 = np.zeros((nq, d_layers), dtype=int)
    k2 = np.zeros((nq, d_layers), dtype=int)
    k3 = np.zeros((nq, d_layers), dtype=int)

    base_sum = int(round(float(seg_sum.mean()))) + 1
    base_mean = int(round(float(seg_mean.mean()))) + 1

    for ell in range(d_layers):
        for i in range(nq):
            k1[i, ell] = (int(round(seg_sum[i])) + ell * base_mean) * int(scale) % BRAID_PERIOD
            k2[i, ell] = (int(round(seg_mean[i])) + (ell + 1) * base_sum) * int(scale) % BRAID_PERIOD
            k3[i, ell] = (int(round(seg_sum[i]) + int(round(seg_mean[(i + 1) % nq]))) + ell) * int(scale) % BRAID_PERIOD
    return k1, k2, k3


# =========================
# Fibonacci braid kolo
# =========================
def build_fibonacci_braid_circuit(
    b_amp: np.ndarray, nq: int, d_layers: int, k1: np.ndarray, k2: np.ndarray, k3: np.ndarray
) -> QuantumCircuit:
    q_reg = QuantumRegister(nq, name="q")
    qc = QuantumCircuit(q_reg)

    qc.append(StatePreparation(b_amp.tolist()), q_reg)

    for ell in range(d_layers):
        for i in range(nq):
            e1 = int(k1[i, ell])
            if e1 > 0:
                sub = QuantumCircuit(1, name=f"s1^{e1}")
                sub.append(UnitaryGate(sigma1_pow(e1), label=f"σ1^{e1}"), [0])
                qc.append(sub.to_gate(label=f"σ1_l{ell}_q{i}"), [q_reg[i]])
            e2 = int(k2[i, ell])
            if e2 > 0:
                sub = QuantumCircuit(1, name=f"s2^{e2}")
                sub.append(UnitaryGate(sigma2_pow(e2), label=f"σ2^{e2}"), [0])
                qc.append(sub.to_gate(label=f"σ2_l{ell}_q{i}"), [q_reg[i]])

        for i in range(nq):
            j = (i + 1) % nq
            e3 = int(k3[i, ell])
            if e3 > 0:
                sub = QuantumCircuit(1, name=f"cs1^{e3}")
                sub.append(UnitaryGate(sigma1_pow(e3), label=f"σ1^{e3}"), [0])
                c_gate = sub.to_gate(label=f"σ1^{e3}").control(num_ctrl_qubits=1)
                qc.append(c_gate, [q_reg[i], q_reg[j]])

    return qc


def fibonacci_braid_probs(
    H: np.ndarray, nq: int, d_layers: int, scale: int
) -> np.ndarray:
    b_amp = amp_from_freq(freq_vector(H), nq)
    k1, k2, k3 = braid_exponents(H, nq, d_layers, scale)
    qc = build_fibonacci_braid_circuit(b_amp, nq, d_layers, k1, k2, k3)
    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    return p


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (D_BRAID, scale)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for d in GRID_D_BRAID:
        for scale in GRID_BRAID_SCALE:
            try:
                p = fibonacci_braid_probs(H, NQ_FIB, int(d), int(scale))
                bi = bias_39(p)
                score = cosine(bi, f_csv_n)
            except Exception:
                continue
            key = (score, int(d), int(scale))
            if best is None or key > best[0]:
                best = (
                    key,
                    dict(d=int(d), scale=int(scale), score=float(score)),
                )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q32 Topological Fibonacci anyon braiding: CSV:", CSV_PATH)
    print(
        "redova:", H.shape[0],
        "| seed:", SEED,
        "| nq_fib:", NQ_FIB,
        "| φ (golden ratio):", round(float(PHI), 10),
    )
    print(
        "--- Fibonacci braid konstante ---",
    )
    print(f"  R_1   = e^(-4πi/5) = {R_1.real:+.6f}{R_1.imag:+.6f}i")
    print(f"  R_τ   = e^( 3πi/5) = {R_TAU.real:+.6f}{R_TAU.imag:+.6f}i")
    print(f"  F²-I  =  {float(np.linalg.norm(F_MAT @ F_MAT - np.eye(2))):.3e}  (treba ≈ 0)")
    print(f"  σ_1^{BRAID_PERIOD} - I = {float(np.linalg.norm(sigma1_pow(BRAID_PERIOD) - np.eye(2))):.3e}  (treba ≈ 0, perioda {BRAID_PERIOD})")

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "D_BRAID=", best["d"],
        "| braid_scale:", best["scale"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    print("--- demonstracija efekta dubine D_BRAID (scale=1) ---")
    for d_demo in GRID_D_BRAID:
        p_d = fibonacci_braid_probs(H, NQ_FIB, int(d_demo), 1)
        pred_d = pick_next_combination(p_d)
        cos_d = cosine(bias_39(p_d), f_csv_n)
        print(f"  D={d_demo:d}  cos={cos_d:.6f}  NEXT={pred_d}")

    p = fibonacci_braid_probs(H, NQ_FIB, int(best["d"]), int(best["scale"]))
    pred = pick_next_combination(p)
    print("--- glavna predikcija (Fibonacci anyon braid-word) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q32 Topological Fibonacci anyon braiding: CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | nq_fib: 6 | φ (golden ratio): 1.6180339887
--- Fibonacci braid konstante ---
  R_1   = e^(-4πi/5) = -0.809017-0.587785i
  R_τ   = e^( 3πi/5) = -0.309017+0.951057i
  F²-I  =  1.614e-16  (treba ≈ 0)
  σ_1^10 - I = 0.000e+00  (treba ≈ 0, perioda 10)
BEST hparam: D_BRAID= 4 | braid_scale: 1 | cos(bias, freq_csv): 0.783003
--- demonstracija efekta dubine D_BRAID (scale=1) ---
  D=1  cos=0.726810  NEXT=(13, 16, 20, 21, 24, 25, 28)
  D=2  cos=0.755585  NEXT=(1, 2, 5, 8, 9, 14, 33)
  D=3  cos=0.701369  NEXT=(3, 5, 7, 9, 13, 15, 25)
  D=4  cos=0.783003  NEXT=(3, 7, 15, 19, 20, 22, 25)
--- glavna predikcija (Fibonacci anyon braid-word) ---
predikcija NEXT: (3, 7, 15, 19, 20, 22, 25)
"""



"""
Q32_Topological_Fibonacci.py — Topological Quantum Computing preko Fibonacci
anyon braid-wordova sa tačnim R i F matricama.

Koncept:
Fibonacci anyoni su univerzalni topološki model kvantnog računanja (non-Abelian
anyoni sa fuzionim pravilom τ x τ = 1 ⊕ τ). Braid operacije σ_1 i σ_2 na jednom
"Fibonacci qubit-u" su 2x2 unitari konstruisani iz jezgrenih topoloških konstanti:
R_1 = e^(-4πi/5), R_τ = e^(3πi/5), F = [[1/φ, 1/√φ], [1/√φ, -1/φ]] gde je φ =
(1+√5)/2 golden ratio. [σ_1, σ_2] generiraju gustu podgrupu SU(2) (univerzalnost
single-qubit-a iz same topologije — Freedman-Kitaev-Larsen-Wang 2003).

Kolo (NQ_FIB qubit-a, BEZ ancilla, BEZ phase-registra):
  StatePreparation(amp_from_freq(freq_csv)) na q-registar.
  Za svaki sloj ℓ ∈ [0, D_BRAID):
    za svaki qubit i: UnitaryGate(σ_1^{k1[i,ℓ]}) zatim UnitaryGate(σ_2^{k2[i,ℓ]}).
    za svaki susedni par (i, i+1) po ringu:
      controlled-UnitaryGate(σ_1^{k3[i,ℓ]}) sa ctrl=i, target=i+1.

Deterministički braid eksponenti iz CSV-a:
  k1[i, ℓ] = (round(seg_sum[i]) + ℓ * base_mean) * scale        mod BRAID_PERIOD=10
  k2[i, ℓ] = (round(seg_mean[i]) + (ℓ+1) * base_sum) * scale    mod 10
  k3[i, ℓ] = (round(seg_sum[i] + seg_mean[i+1]) + ℓ) * scale    mod 10
  gde su seg_sum, seg_mean freq-statistike po segmentu (N_MAX podeljen na nq blokova).

Periodičnost:
  R_τ^10 = 1, R_1^5 = 1 → σ_1^10 = I. Svi eksponenti modulo 10.

Readout:
  Amplituda u computational bazi iz Statevector-a → bias_39 → TOP-7 = NEXT.

Tehnike:
Tačne Fibonacci R_1, R_τ, F konstante (golden ratio).
σ_1 = diag(R_1, R_τ), σ_2 = F σ_1 F (F self-inverz: F² = I).
Multi-qubit entanglement preko controlled-σ_1 (topologija-inspirisan cross-braid).
Periodičnost braid-eksponenata modulo 10 (usklađeno sa R_τ^10 = 1).
UnitaryGate(sigma^k) direktno u qiskit kolu.
Deterministička grid-optimizacija (D_BRAID, scale).

Prednosti:
Tačne Fibonacci matrice (nisu aproksimacije) — prava topološka kvantna struktura.
Univerzalne braid-operacije na single-qubit nivou ({σ_1, σ_2} generiraju SU(2)).
Mali qubit budget (NQ_FIB = 6, bez ancilla, bez phase).
Egzaktni Statevector (bez uzorkovanja).
Ceo CSV (pravilo 10): amplitude i braid eksponenti iz CELOG CSV-a.

Nedostaci:
Inter-qubit braid implementiran kao controlled-σ_1 je topologija-inspirisan
  (nije egzaktan 4x4 Fibonacci anyon cross-braid sa pune 6-anyonske fuzione
  strukture) — egzaktna verzija zahteva eksplicitnu F-move matricu na kombinovanom
  6-anyon prostoru dim F_5 = 5.
Embedovanje u 2^nq bazu (64 za nq=6) može imati nedostupna fuziona stanja; mod-39
  readout se nosi s tim standardnom agregacijom.
Braid eksponenti diskretizovani modulo 10 — gubitak fine-tune kontinuiteta, ali
  to je priroda topološkog računanja (diskretni braid-wordovi).
"""



"""
Topological Quantum Computing preko Fibonacci anyon braid-wordova. 
Koristi tačne topološke konstante: 
R_1 = e^(-4πi/5), R_τ = e^(3πi/5), F-matrica iz golden ratio φ = (1+√5)/2. 
Single-qubit braidovi σ_1 = diag(R_1, R_τ), σ_2 = F·σ_1·F (sa verifikacijom F² = I). 
Multi-qubit entanglement preko controlled-σ_1. 
Eksponenti deterministički iz CSV segmenata, modulo 10 (Fibonacci period). 
Fundamentalno druga klasa od svih Q1-Q31 (TQFT umesto Euclidske dinamike). 


Topological Quantum Computing — Fibonacci anyon braiding Enkodira brojeve kao world-lines Fibonacci anyona na 2D površini. 
Kompjutovanje = braiding (pletenje) anyon putanja — unitar zavisi samo od topologije preplitanja, ne od lokalnih detalja. 
Merenje fuzije anyona daje predikciju. Nova grana: topological quantum field theory.
"""
