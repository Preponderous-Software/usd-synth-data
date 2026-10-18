"""Capture Barony game states from a running Barony backend, over its own HTTP API.

Barony's backend serves the whole game state as JSON (``GET /api/session/state``: the tile
grid with each tile's type and owner, and every army with its position, size and owner). This
script plays guest games against the backend's AI and saves that JSON at chosen ticks, so the
Barony example has boards at different stages of a game. Barony is only called, never
modified or imported; nothing but the standard library is used.

    # in a Barony checkout: build and start the backend (guest mode needs no UserAuth)
    (cd backend && mvn -q package -DskipTests && \\
        USAGE_REPORTING_ENABLED=false java -jar target/backend-*.jar --server.port=18089)
    python examples/barony/capture_states.py --url http://localhost:18089 --games 3 \\
        --ticks 0 15 40 80 --seed 7 --out out/barony-states

Each game is a fresh guest session, so its map comes from the backend's own generator (which
is not seeded through the API: the saved JSON, not the seed, is what makes a run repeatable).
Player 1 (the human side) is driven by this script: every few ticks each idle army is sent to a
random tile it does not own, preferring villages and the enemy castle, and large armies are
sometimes split, so the boards show armies on the move. Player 2 is Barony's own AI.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import random
import urllib.request
from pathlib import Path

HEADERS = {"Content-Type": "application/json", "X-Barony-Player": "guest"}


class Session:
    """One guest game: its own cookie jar, so every Session is a new game."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self.post("/api/guest")

    def _call(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=HEADERS)
        with self.opener.open(req, timeout=30) as resp:
            return json.loads(resp.read() or b"null")

    def post(self, path: str, body=None):
        return self._call("POST", path, body if body is not None else {})

    def state(self) -> dict:
        return self._call("GET", "/api/session/state")


def player_moves(state: dict, rng: random.Random) -> list[dict]:
    """MOVE (and sometimes SPLIT) commands for player 1's idle armies."""
    grid = state["grid"]
    w, h = len(grid), len(grid[0])
    wanted = [(x, y) for x in range(w) for y in range(h)
              if grid[x][y]["type"] != "EMPTY" and grid[x][y]["ownerId"] != 1]  # fmt: skip
    commands = []
    for army in state["armies"]:
        if army["playerId"] != 1 or army.get("moving"):
            continue
        if army["soldiers"] >= 16 and rng.random() < 0.3:
            commands.append({"type": "SPLIT", "armyId": army["id"],
                             "splitAmount": army["soldiers"] // 2})  # fmt: skip
        if wanted and rng.random() < 0.7:
            tx, ty = rng.choice(wanted)
        else:
            tx, ty = rng.randrange(w), rng.randrange(h)
        commands.append({"type": "MOVE", "armyId": army["id"], "targetX": tx, "targetY": ty})
    return commands


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--ticks", type=int, nargs="+", default=[0, 15, 40, 80],
                    help="save the state after these ticks")  # fmt: skip
    ap.add_argument("--command-every", type=int, default=4, help="ticks between player orders")
    ap.add_argument("--seed", type=int, default=7, help="seeds player 1's orders")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    last = max(args.ticks)
    for g in range(args.games):
        game = Session(args.url)
        state = game.state()
        for tick in range(last + 1):
            if tick in args.ticks:
                path = args.out / f"game_{g:02d}_tick_{tick:03d}.json"
                path.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
                armies = len(state["armies"])
                print(f"{path}: {len(state['grid'])}×{len(state['grid'][0])} tiles, "
                      f"{armies} armies, game over: {state['gameOver']}")  # fmt: skip
            if tick == last:
                break
            if tick % args.command_every == 0:
                for command in player_moves(state, rng):
                    game.post("/api/session/command", command)
            state = game.post("/api/session/tick")


if __name__ == "__main__":
    main()
