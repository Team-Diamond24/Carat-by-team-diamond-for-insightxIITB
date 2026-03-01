"""
planner.py - AST-based Code-to-Plan Converter for Safe Pandas Execution

Converts pandas code strings into structured execution plan dictionaries.
Uses python's built-in `ast` module to safely parse and extract operations.
"""

import ast
from typing import Any, Dict, Optional, List, Union

_ALLOWED_AGG_FUNCS = {
    "sum", "mean", "count", "min", "max",
    "median", "std", "var", "nunique", "first", "last", "size",
}

class PandasPlanVisitor(ast.NodeVisitor):
    def __init__(self):
        self.plan: Optional[Dict[str, Any]] = None

    def _extract_string_or_list(self, node: ast.AST) -> Union[str, List[str], None]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            return [self._extract_string_or_list(elt) for elt in node.elts if isinstance(elt, ast.Constant)]
        return None

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        
    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Attribute):
            self.generic_visit(node)
            return

        attr = node.func.attr
        
        if attr in _ALLOWED_AGG_FUNCS:
            self._handle_agg_func(node, attr)
        elif attr == "agg":
            self._handle_dot_agg(node)
        elif attr == "value_counts":
            self._handle_value_counts(node)
        else:
            self.generic_visit(node)

    def _handle_agg_func(self, node: ast.Call, agg_func: str):
        # 1. df.groupby('col')['agg_col'].mean()
        if isinstance(node.func.value, ast.Subscript):
            subscript = node.func.value
            if isinstance(subscript.value, ast.Call) and isinstance(subscript.value.func, ast.Attribute):
                if subscript.value.func.attr == "groupby":
                    self._parse_groupby_bracket_agg(subscript, subscript.value, agg_func)
                    return
        # 2. df['col'].mean()
        elif isinstance(node.func.value, ast.Subscript):
             if isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "df":
                  col = self._extract_string_or_list(node.func.value.slice)
                  if isinstance(col, str):
                       self.plan = {"op": agg_func, "column": col}
                       return
        # 3. df.col.mean()
        elif isinstance(node.func.value, ast.Attribute):
             if getattr(node.func.value.value, "id", "") == "df":
                  col = node.func.value.attr
                  if col not in ("groupby", "agg", "merge", "join", "apply", "reset_index"):
                      self.plan = {"op": agg_func, "column": col}
                      return
        self.generic_visit(node)

    def _parse_groupby_bracket_agg(self, subscript: ast.Subscript, groupby_call: ast.Call, agg_func: str):
         # subscript.value is df.groupby(by_cols)
         if isinstance(groupby_call.func.value, ast.Name) and groupby_call.func.value.id == "df":
             if groupby_call.args:
                  by_cols = self._extract_string_or_list(groupby_call.args[0])
                  if isinstance(by_cols, str): by_cols = [by_cols]
                  agg_cols = self._extract_string_or_list(subscript.slice)
                  if isinstance(agg_cols, str): agg_cols = [agg_cols]
                  
                  if by_cols and agg_cols:
                       self.plan = {"op": "groupby", "by": by_cols, "agg": {c: agg_func for c in agg_cols}}

    def _handle_dot_agg(self, node: ast.Call):
        # df.groupby(...).agg({'col': 'func'})
        call_node = node.func.value
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Attribute):
            if call_node.func.attr == "groupby" and isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "df":
                 if call_node.args and node.args:
                      by_cols = self._extract_string_or_list(call_node.args[0])
                      if isinstance(by_cols, str): by_cols = [by_cols]
                      
                      agg_dict_node = node.args[0]
                      if isinstance(agg_dict_node, ast.Dict):
                          agg = {}
                          for k, v in zip(agg_dict_node.keys, agg_dict_node.values):
                               if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                   if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                       if v.value in _ALLOWED_AGG_FUNCS:
                                           agg[k.value] = v.value
                          
                          if by_cols and agg:
                               self.plan = {"op": "groupby", "by": by_cols, "agg": agg}

    def _handle_value_counts(self, node: ast.Call):
        # df['col'].value_counts()
        if isinstance(node.func.value, ast.Subscript):
             if getattr(node.func.value.value, "id", "") == "df":
                  col = self._extract_string_or_list(node.func.value.slice)
                  if isinstance(col, str):
                      self.plan = {"op": "groupby", "by": [col], "agg": {col: "count"}}

    def visit_Subscript(self, node: ast.Subscript):
         if isinstance(node.value, ast.Name) and node.value.id == "df":
             # Could be df[df['col'] == val] or df[(df['col'] == val) & ...]
             filters = self._extract_filters(node.slice)
             if filters:
                 # BUG 2 FIX: Only set self.plan if it's None
                 if self.plan is None:
                     if len(filters) == 1:
                         self.plan = dict(op="filter", **filters[0])
                     else:
                         self.plan = {"op": "filter_multi", "filters": filters}
                 return
         
         self.generic_visit(node)

    def _extract_filters(self, node: ast.AST) -> List[Dict[str, Any]]:
         filters = []
         
         def _walk(n):
              if isinstance(n, ast.Compare):
                   if isinstance(n.ops[0], ast.Eq):
                        if isinstance(n.left, ast.Subscript) and getattr(n.left.value, "id", "") == "df":
                             col = self._extract_string_or_list(n.left.slice)
                             if isinstance(col, str) and isinstance(n.comparators[0], ast.Constant):
                                  filters.append({"column": col, "value": n.comparators[0].value})
              elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
                   _walk(n.left)
                   _walk(n.right)
                   
         _walk(node)
         return filters

def code_to_plan(code: str) -> Dict[str, Any]:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
        
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        raise ValueError(f"Invalid Python syntax: {code}")
        
    visitor = PandasPlanVisitor()
    visitor.visit(tree)
    
    if visitor.plan:
        return visitor.plan
        
    raise ValueError(
        f"Could not parse code into a supported plan.\\n"
        f"Code: {code}\\n"
        f"Supported AST patterns:\\n"
        f"  - df.groupby('col')['agg_col'].func()\\n"
        f"  - df.groupby('col').agg({{'col': 'func'}})\\n"
        f"  - df['column'].func()\\n"
        f"  - df[df['col'] == value]\\n"
        f"  - df[(df['col1'] == v1) & (df['col2'] == v2)]\\n"
        f"  - df['col'].value_counts()"
    )
