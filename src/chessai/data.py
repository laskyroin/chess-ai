import io
import chess.pgn
import zstandard
from itertools import islice
from pathlib import Path
import numpy as np
from chessai.encoding import board_to_tensor, move_to_index, index_to_move


RESULT_TO_VALUE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

SKIP_OPENING = 10       # demi-coups sautés en début de partie
SAMPLES_PER_GAME = 25   # positions gardées par partie
ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "data" / "lichess_db_standard_rated_2016-02.pgn"
seuilElo = 1900
MIN_BASE_SECONDS = 180

def iter_games(path):
    path = Path(path)
    if path.suffix == ".zst":
        fh = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        stream = open(path, encoding="utf-8", errors="replace")
    with stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                return
            yield game




def get_elo(headers, key):
    v = headers.get(key, "")
    return int(v) if v.isdigit() else None

def get_base_seconds(headers):
    tc = headers.get("TimeControl", "")
    if tc in ("", "-", "?"):
        return None
    base = tc.split("+")[0]
    return int(base) if base.isdigit() else None

def keep_game(game):
    white_elo = get_elo(game.headers, "WhiteElo")
    black_elo = get_elo(game.headers, "BlackElo")
    if white_elo is None or black_elo is None:
        return False
    if min(white_elo, black_elo) < seuilElo:
        return False

    base = get_base_seconds(game.headers)
    if base is None or base < MIN_BASE_SECONDS:
        return False

    if game.headers.get("Termination") != "Normal":
        return False

    return game.headers.get("Result") in RESULT_TO_VALUE

def extract_positions(game, rng):
    value = RESULT_TO_VALUE.get(game.headers.get("Result"))
    if value is None:
        return

    moves = list(game.mainline_moves())
    n = len(moves)
    if n <= SKIP_OPENING:
        return

    candidates = np.arange(SKIP_OPENING, n)
    if len(candidates) > SAMPLES_PER_GAME:
        candidates = rng.choice(candidates, SAMPLES_PER_GAME, replace=False)
    wanted = set(int(i) for i in candidates)

    board = game.board()
    for i, move in enumerate(moves):
        if i in wanted:
            tensor = board_to_tensor(board).numpy().astype(np.uint8)
            yield tensor, move_to_index(move), value
        board.push(move)


def build_dataset(path, out_dir, max_games=None, seed=0):
    rng = np.random.default_rng(seed)
    xs, ps, vs = [], [], []

    games = iter_games(path)
    if max_games is not None:
        games = islice(games, max_games)

    for total, game in enumerate(games):
        if total % 10000 == 0:
            print(total, len(xs))
        if not keep_game(game):
            continue
        for tensor, idx, value in extract_positions(game, rng):
            xs.append(tensor)
            ps.append(idx)
            vs.append(value)
        

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", np.stack(xs))
    np.save(out_dir / "policy.npy", np.array(ps, dtype=np.int16))
    np.save(out_dir / "value.npy", np.array(vs, dtype=np.int8))
    print(len(xs), "positions")


build_dataset(path, "data/full")