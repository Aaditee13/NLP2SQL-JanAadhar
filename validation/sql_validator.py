from __future__ import annotations

from dataclasses import dataclass, field
import sqlglot
from sqlglot import exp

from database.schema_metadata import all_table_names, columns_by_table


DISALLOWED = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "replace",
    "merge",
    "grant",
    "revoke",
    "vacuum",
    "attach",
    "detach",
    "pragma",
    "analyze",
    "reindex",
    "into",
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class SQLValidator:
    def __init__(self):
        self.tables = all_table_names()
        self.columns = columns_by_table()

    def validate(
        self,
        sql: str,
        allowed_tables: list[str] | None = None,
        allowed_columns: list[str] | None = None,
    ) -> ValidationResult:
        errors: list[str] = []
        raw_sql = sql.strip().rstrip(";")

        # Try to parse the SQL using sqlglot
        try:
            # We target SQLite
            tree = sqlglot.parse_one(raw_sql, read="sqlite")
        except Exception as exc:
            return ValidationResult(False, [f"SQL syntax error: {exc}"])

        # ── Check SELECT only ─────────────────────────────────────────────────
        if not isinstance(tree, exp.Select):
            errors.append("Only SELECT statements are allowed.")

        # ── Check disallowed keywords ─────────────────────────────────────────
        for node in tree.walk():
            node_class_name = node.__class__.__name__.lower()
            if any(dw in node_class_name for dw in DISALLOWED):
                errors.append(f"Statement contains disallowed command: {node.__class__.__name__}")
                break

        # ── Check Tables ──────────────────────────────────────────────────────
        allowed_t = {t.lower() for t in (allowed_tables or self.tables)}
        referenced_tables = set()
        
        for table in tree.find_all(exp.Table):
            t_name = table.name.lower()
            if t_name:
                referenced_tables.add(t_name)
                if t_name not in allowed_t:
                    errors.append(f"Unknown/disallowed table referenced: {table.name}")

        # ── Check Columns ─────────────────────────────────────────────────────
        allowed_c = {c.lower() for c in (allowed_columns or [])}
        for t in (allowed_tables or self.tables):
            allowed_c.update(c.lower() for c in self.columns.get(t, []))
            
        # Extract aliases to allow them in outer clauses
        for alias_node in tree.find_all(exp.Alias):
            if alias_node.alias:
                allowed_c.add(alias_node.alias.lower())

        _ALWAYS_ALLOWED = {"*", "1", "current_date", "current_timestamp", "count"}
        
        for col_node in tree.find_all(exp.Column):
            col_name = col_node.name.lower() if col_node.name else ""
            if col_name and col_name not in allowed_c and col_name not in _ALWAYS_ALLOWED:
                errors.append(f"Unknown/disallowed column referenced: {col_node.name}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)
