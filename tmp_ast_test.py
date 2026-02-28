import ast
from typing import Dict, Any

class PandasPlanVisitor(ast.NodeVisitor):
    def __init__(self):
        self.plan = None

    def _extract_string_or_list(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.List):
            return [self._extract_string_or_list(elt) for elt in node.elts]
        return None

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)

    def visit_Call(self, node: ast.Call):
        # We handle chain calls from outside in
        # df.groupby(...)[...].agg(...)
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in {"sum", "mean", "count", "min", "max", "median", "std", "var", "nunique", "first", "last", "size"}:
                # Pattern: something.agg()
                if isinstance(node.func.value, ast.Subscript):
                    # df.groupby(...)[agg_col].mean()
                    self._parse_groupby_bracket_agg(node, attr)
                    return
                elif isinstance(node.func.value, ast.Attribute) and getattr(node.func.value.value, "id", "") == "df":
                    # df.column.mean()
                    self.plan = {"op": attr, "column": node.func.value.attr}
                    return
                elif getattr(node.func.value, "value", None):
                   pass # more complex

            elif attr == "agg":
                # Pattern: df.groupby(...).agg({'col': 'func'})
                self._parse_groupby_dot_agg(node)
                return
            elif attr == "value_counts":
                self._parse_value_counts(node)
                return
            else:
                self.visit(node.func.value)

    def _parse_value_counts(self, node: ast.Call):
        # df[col].value_counts()
        if isinstance(node.func.value, ast.Subscript):
            if isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "df":
                 idx_node = node.func.value.slice
                 col = self._extract_string_or_list(idx_node)
                 if col:
                     self.plan = {"op": "groupby", "by": [col], "agg": {col: "count"}}

    def _parse_groupby_bracket_agg(self, node: ast.Call, agg_func: str):
         # df.groupby(by_cols)[agg_col].func()
         subscript_node = node.func.value
         if isinstance(subscript_node.value, ast.Call):
             call_node = subscript_node.value
             if isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "groupby":
                 if isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "df":
                     if call_node.args:
                         by_cols = self._extract_string_or_list(call_node.args[0])
                         if isinstance(by_cols, str): by_cols = [by_cols]
                         agg_cols = self._extract_string_or_list(subscript_node.slice)
                         if by_cols and agg_cols:
                             self.plan = {"op": "groupby", "by": by_cols, "agg": {agg_cols: agg_func}}

    def _parse_groupby_dot_agg(self, node: ast.Call):
         # df.groupby(by_cols).agg({'col': 'func'})
         call_node = node.func.value
         if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "groupby":
             if isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "df":
                  if call_node.args and node.args:
                       by_cols = self._extract_string_or_list(call_node.args[0])
                       if isinstance(by_cols, str): by_cols = [by_cols]
                       agg_dict_node = node.args[0]
                       if isinstance(agg_dict_node, ast.Dict):
                            agg = {}
                            for k, v in zip(agg_dict_node.keys, agg_dict_node.values):
                                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                                    agg[k.value] = v.value
                            self.plan = {"op": "groupby", "by": by_cols, "agg": agg}

    def visit_Subscript(self, node: ast.Subscript):
         # Handle filter: df[condition] or df['col'] agg
         if isinstance(node.value, ast.Name) and node.value.id == "df":
             if isinstance(node.slice, ast.Compare):
                  # df[df['col'] == val]
                  self._parse_filter_compare(node.slice)
             elif isinstance(node.slice, ast.BoolOp):
                  # df[(df['col'] == val) & (df['col2'] == val2)] -> need complex filter struct
                  pass
             elif isinstance(node.slice, ast.BinOp):
                  # & is a BitAnd over boolean masks
                  self._parse_binop_filter(node.slice)
             else:
                  # maybe df['col'].mean() where parent is Subscript... wait that's a Call wrapping an Attribute wrapping a Subscript.
                  # visit_Call handled it.
                  self.generic_visit(node)
         else:
             self.generic_visit(node)

    def _parse_filter_compare(self, node: ast.Compare):
         # df['col'] == val
         if isinstance(node.ops[0], ast.Eq):
              if isinstance(node.left, ast.Subscript) and getattr(node.left.value, "id", "") == "df":
                   col = self._extract_string_or_list(node.left.slice)
                   if isinstance(node.comparators[0], ast.Constant):
                        val = node.comparators[0].value
                        self.plan = {"op": "filter", "column": col, "value": val}

    def _parse_binop_filter(self, node: ast.BinOp):
        # We need to extract multiple filters. For safe_exec, maybe just pass complex filters? Or implement "filter_multi"?
        # For now, let's just parse it directly to see if we can extract elements.
        filters = []
        def _walk_binop(op_node):
             if isinstance(op_node, ast.BinOp) and isinstance(op_node.op, ast.BitAnd):
                 _walk_binop(op_node.left)
                 _walk_binop(op_node.right)
             elif isinstance(op_node, ast.Compare):
                 if isinstance(op_node.ops[0], ast.Eq):
                      col = self._extract_string_or_list(op_node.left.slice)
                      val = op_node.comparators[0].value
                      filters.append({"column": col, "value": val})
        _walk_binop(node)
        if filters:
             self.plan = {"op": "filter_multi", "filters": filters}


def code_to_plan(code: str):
    tree = ast.parse(code)
    visitor = PandasPlanVisitor()
    visitor.visit(tree)
    return visitor.plan

print('1', code_to_plan("df.groupby('sender_bank')['amount_inr'].mean()"))
print('2', code_to_plan("df.groupby(['sender_bank', 'state'])['amount_inr'].sum()"))
print('3', code_to_plan("df.groupby('sender_state').agg({'amount_inr': 'sum', 'fraud_flag': 'mean'})"))
print('4', code_to_plan("df['amount_inr'].mean()"))
print('5', code_to_plan("result = df[df['transaction_status'] == 'FAILED']"))
print('6', code_to_plan("result = df['device_type'].value_counts()"))
print('7', code_to_plan("result = df[(df['transaction_status'] == 'FAILED') & (df['device_type'] == 'ANDROID')]"))
