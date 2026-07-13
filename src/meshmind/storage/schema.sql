-- MeshMind SQLite schema.
--
-- Design notes:
--   * Hyperedges are FIRST-CLASS. They are not (head, relation, tail) rows.
--     A hyperedge lives in `hyperedges`; its participants live in
--     `hyperedge_nodes`, one row per (edge, node) with a role + weight. This is
--     what lets a single edge connect arbitrarily many nodes (arity >= 2).
--   * Embeddings are stored as raw float32 BLOBs and searched with numpy in
--     Python. We deliberately avoid sqlite-vss: keeping vectors as plain blobs
--     means a MeshMind .db is a single portable file with zero native
--     extensions required. See DESIGN.md ("Storage") for the trade-off.
--   * FTS5 gives us cheap lexical seed-finding for spreading activation even
--     before/without semantic embeddings.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Nodes: memory units.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS nodes (
    id           TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'fact',
    confidence   REAL NOT NULL DEFAULT 1.0,
    decay_rate   REAL NOT NULL DEFAULT 0.05,
    created_at   REAL NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'   -- JSON
);

-- ---------------------------------------------------------------------------
-- Hyperedges: first-class N-ary relations.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hyperedges (
    id                TEXT PRIMARY KEY,
    type              TEXT NOT NULL,
    activation_weight REAL NOT NULL DEFAULT 1.0,
    decay_rate        REAL NOT NULL DEFAULT 0.03,
    confidence        REAL NOT NULL DEFAULT 1.0,
    created_at        REAL NOT NULL,
    provenance        TEXT NOT NULL DEFAULT '{}',  -- JSON
    metadata          TEXT NOT NULL DEFAULT '{}'   -- JSON
);

-- ---------------------------------------------------------------------------
-- The join table that makes hyperedges real. Arbitrary arity: N rows here per
-- hyperedge. `role` = how the node participates; `weight` = strength of the
-- node<->edge coupling for activation flow.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hyperedge_nodes (
    hyperedge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
    node_id      TEXT NOT NULL REFERENCES nodes(id)      ON DELETE CASCADE,
    role         TEXT NOT NULL DEFAULT 'member',
    weight       REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (hyperedge_id, node_id)
);

CREATE INDEX IF NOT EXISTS idx_hn_node ON hyperedge_nodes(node_id);
CREATE INDEX IF NOT EXISTS idx_hn_edge ON hyperedge_nodes(hyperedge_id);
CREATE INDEX IF NOT EXISTS idx_edge_type ON hyperedges(type);

-- ---------------------------------------------------------------------------
-- Activations: the time-decayed salience of a node. Kept in its own table so
-- reinforcement/decay updates never rewrite the (immutable) node content, and
-- so history can be reconstructed if desired.
--   base      = activation value recorded at `updated_at`
--   updated_at= when that value was written (decay is applied from here)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS activations (
    node_id    TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    base       REAL NOT NULL DEFAULT 1.0,
    updated_at REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0
);

-- ---------------------------------------------------------------------------
-- Embeddings: float32 vectors stored as BLOBs, searched in numpy.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embeddings (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    dim     INTEGER NOT NULL,
    vector  BLOB NOT NULL           -- little-endian float32
);

-- ---------------------------------------------------------------------------
-- Full-text index over node text for lexical seed discovery.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    text,
    content='nodes',
    content_rowid='rowid'
);

-- Keep the FTS index in sync with `nodes` via triggers.
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO nodes_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Schema version marker for the portable format / migrations.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');
