from pathlib import Path
import sqlite3
from dataclasses import dataclass
from ..symbolic.symbolic_state_emitter import SymbolicState
import json

@dataclass
class MemoryVault:
    path: Path

    def __post_init__(self):
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            'CREATE TABLE IF NOT EXISTS states (hash TEXT PRIMARY KEY, data TEXT)'
        )
        self.conn.execute(
            'CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)'
        )

    def checkpoint(self, state: SymbolicState) -> str:
        h = str(abs(hash(tuple(state.predicates))))
        data = json.dumps({'predicates': state.predicates, 'confidence': state.confidence})
        self.conn.execute('INSERT OR REPLACE INTO states (hash, data) VALUES (?, ?)', (h, data))
        self.conn.commit()
        return h

    def retrieve(self, h: str) -> SymbolicState:
        cur = self.conn.execute('SELECT data FROM states WHERE hash = ?', (h,))
        row = cur.fetchone()
        if not row:
            raise KeyError(h)
        data = json.loads(row[0])
        return SymbolicState(predicates=data['predicates'], confidence=data['confidence'])

    def add_knowledge(self, texts: list[str]) -> None:
        for text in texts:
            self.conn.execute('INSERT INTO knowledge (text) VALUES (?)', (text,))
        self.conn.commit()
