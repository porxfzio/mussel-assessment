CREATE TABLE sessions (
    id               TEXT PRIMARY KEY,
    stage            INTEGER DEFAULT 1,
    shell_a_path     TEXT,
    shell_b_path     TEXT,
    meat_path        TEXT,
    initial_grade    TEXT,
    final_grade      TEXT,
    initial_features TEXT,
    final_features   TEXT,
    created_at       TIMESTAMP DEFAULT NOW()
);