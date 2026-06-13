import sqlglot
import sqlglot.expressions as exp
from typing import Optional
from nlp.entity_extractor import EntityExtractor, ExtractionResult, Condition

# Numeric comparison operators supported for both generation and swapping
_NUMERIC_OPS = (">", "<", ">=", "<=")
_SWAPPABLE_OPS = ("=", "LIKE") + _NUMERIC_OPS


class FastPathEngine:
    def __init__(self):
        self.extractor = EntityExtractor()

    def generate_sql_fast(self, question: str) -> Optional[str]:
        """
        Tier 0: Generates SQL instantly for simple, fully-understood queries.
        Returns None if the query is too complex — defers to Semantic Cache / LLM.
        """
        res = self.extractor.extract(question)

        # Abort if complex keywords detected or any nouns were not understood
        if res.has_complex_keywords or len(res.unhandled_words) > 0:
            return None

        if not res.entities:
            return None

        # Safety cap: >3 distinct filter columns → defer to LLM
        if len(res.entities) > 3:
            return None

        conditions = []
        for col, conds in res.entities.items():
            for cond in conds:
                node = self._build_condition(col, cond)
                if node is not None:
                    conditions.append(node)

        if not conditions:
            return None

        where_clause = exp.and_(*conditions) if len(conditions) > 1 else conditions[0]
        select_cols = [
            exp.column("name_en"),
            exp.column("age"),
            exp.column("gender"),
            exp.column("district_name_eng"),
        ]
        query = exp.select(*select_cols).from_("citizen").where(where_clause)
        return query.sql(dialect="sqlite") + ";"
 
    # ── Private: build one sqlglot condition node ─────────────────────────────
    def _build_condition(self, col: str, cond: Condition) -> Optional[exp.Expression]:
        col_exp = exp.column(col)

        if cond.operator == "=":
            if isinstance(cond.value, int):
                return exp.EQ(this=col_exp, expression=exp.Literal.number(cond.value))

            # ── Education special handling (mirrors app.py _post_process_sql) ──
            if col == "education":
                if cond.value == "illiterate":
                    # DB stores lowercase 'illiterate'; wrap in LOWER() for safety
                    return exp.EQ(
                        this=exp.Lower(this=col_exp),
                        expression=exp.Literal.string("illiterate"),
                    )
                # All other education values → LIKE for partial/case-insensitive match
                return exp.Like(
                    this=col_exp,
                    expression=exp.Literal.string(f"%{cond.value}%"),
                )

            # ── bank_name → UPPER(col) LIKE '%VALUE%' ────────────────────────
            if col == "bank":
                return exp.Like(
                    this=exp.Upper(this=col_exp),
                    expression=exp.Literal.string(f"%{cond.value.upper()}%"),
                )

            return exp.EQ(this=col_exp, expression=exp.Literal.string(cond.value))

        elif cond.operator in _NUMERIC_OPS:
            val_exp = exp.Literal.number(cond.value)
            op_map = {
                ">":  exp.GT,
                "<":  exp.LT,
                ">=": exp.GTE,
                "<=": exp.LTE,
            }
            return op_map[cond.operator](this=col_exp, expression=val_exp)

        elif cond.operator == "BETWEEN":
            return exp.Between(
                this=col_exp,
                low=exp.Literal.number(cond.value[0]),
                high=exp.Literal.number(cond.value[1]),
            )

        elif cond.operator == "LIKE":
            return exp.Like(
                this=col_exp,
                expression=exp.Literal.string(f"%{cond.value}%"),
            )

        elif cond.operator == "IS NULL":
            return exp.Is(this=col_exp, expression=exp.null())

        elif cond.operator == "IS NOT NULL":
            return exp.Is(this=col_exp, expression=exp.Not(this=exp.null()))

        elif cond.operator == "IN":
            val_list = [
                exp.Literal.string(v) if isinstance(v, str) else exp.Literal.number(v)
                for v in cond.value
            ]
            return exp.In(this=col_exp, expressions=val_list)

        elif cond.operator == "LIKE_ANY":
            likes = [
                exp.Like(this=exp.column(col), expression=exp.Literal.string(f"%{v}%"))
                for v in cond.value
            ]
            if len(likes) == 1:
                return likes[0]
            return exp.or_(*likes)

        return None

    # ── Smart Cache: surgical AST parameter swap ──────────────────────────────
    def swap_ast_parameters(
        self, cached_sql: str, cached_question: str, new_question: str
    ) -> Optional[str]:
        """
        Tier 1.5: Mutates cached SQL at the AST level when only parameter values
        differ between two structurally identical questions.
        Supports string (=, LIKE) and numeric (>, <, >=, <=) operators.
        """
        old_res = self.extractor.extract(cached_question)
        new_res = self.extractor.extract(new_question)

        # Build delta: {col: new_value} for every column whose value changed
        # while keeping the same operator (safe substitution only)
        delta: dict[str, object] = {}
        for col, new_conds in new_res.entities.items():
            old_conds = old_res.entities.get(col, [])
            if len(new_conds) != 1 or len(old_conds) != 1:
                continue
            n, o = new_conds[0], old_conds[0]
            if n.operator == o.operator and n.operator in _SWAPPABLE_OPS:
                if n.value != o.value:
                    delta[col] = n.value

        if not delta:
            return None

        # AST Surgery
        try:
            tree = sqlglot.parse_one(cached_sql, dialect="sqlite")
        except Exception:
            return None

        swap_count = 0

        def transformer(node):
            nonlocal swap_count

            # ── String operators: EQ, Like ────────────────────────────────────
            if isinstance(node, (exp.EQ, exp.Like)):
                left, right = node.left, node.right
                # Also handle LOWER(col) = 'val' pattern (education illiterate)
                actual_col = None
                if isinstance(left, exp.Column):
                    actual_col = left.name.lower()
                elif isinstance(left, exp.Lower) and isinstance(left.this, exp.Column):
                    actual_col = left.this.name.lower()

                if actual_col and actual_col in delta and isinstance(right, exp.Literal):
                    new_val = delta[actual_col]
                    if isinstance(new_val, int):
                        right.set("this", str(new_val))
                    else:
                        # Preserve % wildcards for LIKE patterns
                        right.set("this", f"%{new_val}%" if "%" in right.name else new_val)
                    swap_count += 1

            # ── Numeric operators: GT, LT, GTE, LTE ──────────────────────────
            elif isinstance(node, (exp.GT, exp.LT, exp.GTE, exp.LTE)):
                left, right = node.left, node.right
                if isinstance(left, exp.Column) and left.name.lower() in delta:
                    new_val = delta[left.name.lower()]
                    if isinstance(new_val, (int, float)):
                        right.set("this", str(int(new_val)))
                        swap_count += 1

            return node

        tree = tree.transform(transformer)

        if swap_count > 0:
            return tree.sql(dialect="sqlite") + ";"
        return None
