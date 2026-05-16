"""Python ctypes wrapper for the Odin alpha_go backend.

Mirrors the public surface of the upstream `alpha_go_cpp` pybind11 module
(autogo/src/alpha_go/cpp/bindings/bindings.cpp) so callers can transparently
swap backends.

Currently covered:
    - GoBoard (full)
    - MCTSConfig
    - MCTSTree (single-state callback evaluator; no batched path yet)

Set env var ``ALPHA_GO_ODIN_LIB`` to override the .so path; otherwise the loader
looks for ``build/libalpha_go_odin.so`` relative to the repo root.
"""

from __future__ import annotations

import ctypes as ct
import os
import pathlib
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------------------- #
# .so loader
# --------------------------------------------------------------------------- #

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_DEFAULT_LIB = _REPO_ROOT / "build" / "libalpha_go_odin.so"


def _load_lib() -> ct.CDLL:
    path = os.environ.get("ALPHA_GO_ODIN_LIB", str(_DEFAULT_LIB))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Odin shared lib not found at {path}; run scripts/build_odin.sh."
        )
    return ct.CDLL(path)


_lib = _load_lib()


def _bind(name: str, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


# Constants
PASS_ACTION: int = _bind("alphago_pass_action", ct.c_int, [])()
EMPTY: int = _bind("alphago_empty", ct.c_char, [])()[0]
BLACK: int = _bind("alphago_black", ct.c_char, [])()[0]
WHITE: int = _bind("alphago_white", ct.c_char, [])()[0]
KOMI: float = _bind("alphago_komi_default", ct.c_float, [])()

# Type aliases
_Handle = ct.c_void_p
_PInt = ct.POINTER(ct.c_int)
_PFloat = ct.POINTER(ct.c_float)
_PChar = ct.POINTER(ct.c_char)

# GoBoard bindings
_b_new = _bind("alphago_goboard_new", _Handle, [ct.c_int, ct.c_float])
_b_free = _bind("alphago_goboard_free", None, [_Handle])
_b_copy = _bind("alphago_goboard_copy", _Handle, [_Handle])
_b_size = _bind("alphago_goboard_size", ct.c_int, [_Handle])
_b_to_play = _bind("alphago_goboard_to_play", ct.c_char, [_Handle])
_b_move_count = _bind("alphago_goboard_move_count", ct.c_int, [_Handle])
_b_komi = _bind("alphago_goboard_komi", ct.c_float, [_Handle])
_b_ko_point = _bind("alphago_goboard_ko_point", ct.c_int, [_Handle])
_b_current_hash = _bind("alphago_goboard_current_hash", ct.c_uint64, [_Handle])
_b_at_flat = _bind("alphago_goboard_at_flat", ct.c_char, [_Handle, ct.c_int])
_b_play = _bind("alphago_goboard_play", ct.c_int, [_Handle, ct.c_int, ct.c_int])
_b_play_flat = _bind("alphago_goboard_play_flat", ct.c_int, [_Handle, ct.c_int])
_b_pass = _bind("alphago_goboard_pass", ct.c_int, [_Handle])
_b_is_legal = _bind("alphago_goboard_is_legal", ct.c_int, [_Handle, ct.c_int, ct.c_int])
_b_is_legal_flat = _bind("alphago_goboard_is_legal_flat", ct.c_int, [_Handle, ct.c_int])
_b_is_game_over = _bind("alphago_goboard_is_game_over", ct.c_int, [_Handle])
_b_score = _bind("alphago_goboard_score", ct.c_float, [_Handle])
_b_get_winner = _bind("alphago_goboard_get_winner", ct.c_char, [_Handle])
_b_legal_moves = _bind(
    "alphago_goboard_get_legal_moves_flat", ct.c_int, [_Handle, _PInt, ct.c_int]
)
_b_to_array = _bind("alphago_goboard_to_array", None, [_Handle, _PChar, ct.c_int])
_b_set_from_array = _bind(
    "alphago_goboard_set_from_array", None, [_Handle, _PChar, ct.c_char]
)


def _i8(byte: bytes) -> int:
    """Interpret a single-byte ctypes c_char return as a signed-int color value."""
    return int.from_bytes(byte, byteorder="little", signed=True)


class GoBoard:
    EMPTY = EMPTY
    BLACK = BLACK
    WHITE = WHITE
    KOMI = KOMI

    def __init__(self, size: int = 9, komi: float = KOMI, _handle: _Handle | None = None):
        if _handle is not None:
            self._h = _handle
        else:
            self._h = _b_new(size, komi)
        self._size = size if _handle is None else _b_size(self._h)
        self._owned = True

    def __del__(self):
        try:
            if getattr(self, "_owned", False) and getattr(self, "_h", None):
                _b_free(self._h)
                self._h = None
        except Exception:
            pass

    def size(self) -> int:
        return _b_size(self._h)

    def to_play(self) -> int:
        return _i8(_b_to_play(self._h))

    def move_count(self) -> int:
        return _b_move_count(self._h)

    def komi(self) -> float:
        return _b_komi(self._h)

    def ko_point(self) -> int:
        return _b_ko_point(self._h)

    def current_hash(self) -> int:
        return int(_b_current_hash(self._h))

    def at_flat(self, idx: int) -> int:
        return _i8(_b_at_flat(self._h, idx))

    def at(self, row: int, col: int) -> int:
        return self.at_flat(row * self._size + col)

    def play(self, row: int, col: int) -> bool:
        return _b_play(self._h, row, col) != 0

    def play_flat(self, idx: int) -> bool:
        return _b_play_flat(self._h, idx) != 0

    def pass_move(self) -> bool:
        return _b_pass(self._h) != 0

    def is_legal(self, row: int, col: int) -> bool:
        return _b_is_legal(self._h, row, col) != 0

    def is_legal_flat(self, idx: int) -> bool:
        return _b_is_legal_flat(self._h, idx) != 0

    def is_game_over(self) -> bool:
        return _b_is_game_over(self._h) != 0

    def score(self) -> float:
        return _b_score(self._h)

    def get_winner(self) -> int:
        return _i8(_b_get_winner(self._h))

    def get_legal_moves_flat(self) -> list[int]:
        cap = self._size * self._size + 1
        buf = (ct.c_int * cap)()
        n = _b_legal_moves(self._h, buf, cap)
        return [buf[i] for i in range(n)]

    def to_numpy(self) -> np.ndarray:
        n = self._size * self._size
        buf = (ct.c_char * n)()
        _b_to_array(self._h, buf, n)
        arr = np.frombuffer(bytes(buf), dtype=np.int8).copy()
        return arr.reshape(self._size, self._size)

    def set_from_numpy(self, arr: np.ndarray, to_play: int) -> None:
        if arr.shape != (self._size, self._size):
            raise ValueError(f"shape {arr.shape} != ({self._size}, {self._size})")
        flat = np.ascontiguousarray(arr, dtype=np.int8).reshape(-1)
        buf = flat.ctypes.data_as(_PChar)
        _b_set_from_array(self._h, buf, ct.c_char(bytes([to_play & 0xFF])))

    def copy(self) -> "GoBoard":
        h = _b_copy(self._h)
        return GoBoard(self._size, _handle=h)

    def row_col(self, flat: int) -> tuple[int, int]:
        return (flat // self._size, flat % self._size)

    def __repr__(self) -> str:
        color = "BLACK" if self.to_play() == BLACK else "WHITE"
        return (
            f"GoBoard({self._size}x{self._size}, to_play={color}, "
            f"moves={self.move_count()})"
        )


# --------------------------------------------------------------------------- #
# MCTS
# --------------------------------------------------------------------------- #

_cfg_new = _bind("alphago_mcts_config_new", _Handle, [])
_cfg_free = _bind("alphago_mcts_config_free", None, [_Handle])
_cfg_set = _bind(
    "alphago_mcts_config_set",
    None,
    [
        _Handle,
        ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float,
        ct.c_int,
    ],
)
_cfg_set_pcr = _bind("alphago_mcts_config_set_pcr", None, [_Handle, _PInt, _PFloat, ct.c_int])

_t_new = _bind("alphago_mcts_tree_new", _Handle, [_Handle, _Handle, ct.c_uint64])
_t_free = _bind("alphago_mcts_tree_free", None, [_Handle])
_t_size = _bind("alphago_mcts_tree_size", ct.c_int, [_Handle])
_t_root_visits = _bind("alphago_mcts_tree_root_visits", ct.c_int, [_Handle])
_t_root_q = _bind("alphago_mcts_tree_root_q", ct.c_float, [_Handle])
_t_select_action = _bind("alphago_mcts_tree_select_action", ct.c_int, [_Handle, ct.c_float])
_t_child_visits = _bind(
    "alphago_mcts_tree_child_visits", ct.c_int, [_Handle, _PInt, _PInt, ct.c_int]
)
_t_child_q = _bind(
    "alphago_mcts_tree_child_q_values", ct.c_int, [_Handle, _PInt, _PFloat, ct.c_int]
)
_t_action_probs = _bind(
    "alphago_mcts_tree_action_probabilities",
    ct.c_int,
    [_Handle, ct.c_float, _PInt, _PFloat, ct.c_int],
)

_CEvaluator = ct.CFUNCTYPE(
    ct.c_int,         # n actions written
    ct.c_void_p,      # goboard rawptr (non-owning view; MCTS owns lifetime)
    _PInt,            # out_actions
    _PFloat,          # out_probs
    ct.c_int,         # max_n
    _PFloat,          # out_value
    ct.c_void_p,      # user_data
)
_t_run_sims = _bind(
    "alphago_mcts_tree_run_simulations", None, [_Handle, ct.c_int, _CEvaluator, ct.c_void_p]
)

# Batched evaluator C-ABI. Mirrors the sequential _CEvaluator but flat-shaped:
# batch_size pointers to GoBoard, plus row-major out buffers sized
# batch_size * max_n_per_state.
_CEvaluatorBatched = ct.CFUNCTYPE(
    None,                  # void return — counts/values are written through pointers
    ct.c_int,              # batch_size
    ct.POINTER(ct.c_void_p),  # states[batch_size]
    _PInt,                 # out_actions (flat, row-major)
    _PFloat,               # out_probs   (flat, row-major)
    _PInt,                 # out_counts[batch_size]
    _PFloat,               # out_values[batch_size]
    ct.c_int,              # max_n_per_state
    ct.c_void_p,           # user_data
)
_t_run_sims_batched = _bind(
    "alphago_mcts_tree_run_simulations_batched",
    None,
    [_Handle, ct.c_int, ct.c_int, _CEvaluatorBatched, ct.c_void_p],
)


class MCTSConfig:
    def __init__(self):
        self._h = _cfg_new()
        self.c_puct = 1.0
        self.lambda_ = 0.0
        self.dirichlet_alpha = 0.0
        self.dirichlet_weight = 0.25
        self.temperature = 1.0
        self.max_depth = 100
        self.rollout_temperature = 1.0
        self.pcr_sims: list[int] = []
        self.pcr_probs: list[float] = []

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                _cfg_free(self._h)
                self._h = None
        except Exception:
            pass

    def _sync_to_native(self) -> None:
        _cfg_set(
            self._h,
            self.c_puct,
            self.lambda_,
            self.dirichlet_alpha,
            self.dirichlet_weight,
            self.temperature,
            self.rollout_temperature,
            self.max_depth,
        )
        if self.pcr_sims:
            n = len(self.pcr_sims)
            sims = (ct.c_int * n)(*self.pcr_sims)
            probs = (ct.c_float * n)(*self.pcr_probs)
            _cfg_set_pcr(self._h, sims, probs, n)
        else:
            _cfg_set_pcr(self._h, None, None, 0)


PolicyValue = tuple[dict[int, float], float]
EvaluatorFn = Callable[[GoBoard], PolicyValue]
BatchedEvaluatorFn = Callable[[list[GoBoard]], tuple[list[dict[int, float]], list[float]]]


class MCTSTree:
    def __init__(self, root_state: GoBoard, config: MCTSConfig, seed: int = 0):
        config._sync_to_native()
        self._board_size = root_state.size()
        self._h = _t_new(root_state._h, config._h, ct.c_uint64(seed))
        # Stash the bound CFUNCTYPE so it isn't GC'd mid-call.
        self._cb_keepalive: Any = None

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                _t_free(self._h)
                self._h = None
        except Exception:
            pass

    def tree_size(self) -> int:
        return _t_size(self._h)

    def get_root_visit_count(self) -> int:
        return _t_root_visits(self._h)

    def get_root_q_value(self) -> float:
        return _t_root_q(self._h)

    def select_action(self, temperature: float = 1.0) -> int:
        return _t_select_action(self._h, temperature)

    def _make_trampoline(self, evaluator: EvaluatorFn):
        board_size = self._board_size

        def trampoline(goboard_ptr, out_actions, out_probs, max_n, out_value, _user):
            # Non-owning GoBoard view over the leaf's existing Odin-side board.
            # MCTS owns the lifetime; do NOT let __del__ free it.
            view = GoBoard.__new__(GoBoard)
            view._h = goboard_ptr
            view._owned = False
            view._size = board_size
            policy, value = evaluator(view)
            i = 0
            for action, prob in policy.items():
                if i >= max_n:
                    break
                out_actions[i] = action
                out_probs[i] = prob
                i += 1
            out_value[0] = value
            return i

        return _CEvaluator(trampoline)

    def run_simulations(self, num_simulations: int, evaluator: EvaluatorFn) -> None:
        cb = self._make_trampoline(evaluator)
        # Keep cb alive for the duration of the call so ctypes doesn't free the
        # trampoline mid-flight.
        self._cb_keepalive = cb
        _t_run_sims(self._h, num_simulations, cb, None)
        self._cb_keepalive = None

    def _make_batched_trampoline(self, evaluator: BatchedEvaluatorFn):
        board_size = self._board_size

        def trampoline(batch_size, states_ptr, out_actions, out_probs, out_counts, out_values, max_n, _user):
            # Wrap each state pointer as a non-owning GoBoard view.
            views: list[GoBoard] = []
            for i in range(batch_size):
                view = GoBoard.__new__(GoBoard)
                view._h = states_ptr[i]
                view._owned = False
                view._size = board_size
                views.append(view)
            policies, values = evaluator(views)
            # Pack into the row-major out_actions / out_probs buffers.
            for i, (policy, value) in enumerate(zip(policies, values)):
                row_base = i * max_n
                k = 0
                for action, prob in policy.items():
                    if k >= max_n:
                        break
                    out_actions[row_base + k] = action
                    out_probs[row_base + k]   = prob
                    k += 1
                out_counts[i] = k
                out_values[i] = value

        return _CEvaluatorBatched(trampoline)

    def run_simulations_batched(
        self,
        num_simulations: int,
        batch_size: int,
        evaluator: BatchedEvaluatorFn,
    ) -> None:
        cb = self._make_batched_trampoline(evaluator)
        # Keep cb alive (same pattern as run_simulations).
        self._cb_keepalive = cb
        _t_run_sims_batched(self._h, num_simulations, batch_size, cb, None)
        self._cb_keepalive = None

    def get_child_visit_counts(self) -> dict[int, int]:
        cap = self._board_size * self._board_size + 1
        a = (ct.c_int * cap)()
        c = (ct.c_int * cap)()
        n = _t_child_visits(self._h, a, c, cap)
        return {a[i]: c[i] for i in range(min(n, cap))}

    def get_child_q_values(self) -> dict[int, float]:
        cap = self._board_size * self._board_size + 1
        a = (ct.c_int * cap)()
        q = (ct.c_float * cap)()
        n = _t_child_q(self._h, a, q, cap)
        return {a[i]: q[i] for i in range(min(n, cap))}

    def get_action_probabilities(self, temperature: float = 1.0) -> dict[int, float]:
        cap = self._board_size * self._board_size + 1
        a = (ct.c_int * cap)()
        p = (ct.c_float * cap)()
        n = _t_action_probs(self._h, temperature, a, p, cap)
        return {a[i]: p[i] for i in range(min(n, cap))}


__version__ = "0.0.1-scaffold"
__all__ = [
    "PASS_ACTION", "EMPTY", "BLACK", "WHITE", "KOMI",
    "GoBoard", "MCTSConfig", "MCTSTree",
    "__version__",
]
