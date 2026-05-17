"""
ast_parser/python_parser.py
────────────────────────────
Tree-sitter based Python AST parser.

Extracts from each .py file:
  - Module-level imports (import X, from X import Y)
  - Function definitions: name, args, decorators, line range
  - Class definitions: name, bases, methods, line range
  - Call expressions (who calls whom)
  - Docstrings (for BM25 indexing in Phase 3)

Output is a structured FileSymbols dataclass serialisable to JSON.
Cached per file SHA-256 so repeat queries cost zero re-parse.

Tree-sitter grammar used: tree-sitter-python
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ImportInfo:
    module: str          # the module being imported
    names: list[str]     # specific names imported (empty = wildcard/module)
    is_from: bool        # True for 'from X import Y', False for 'import X'
    alias: str = ""      # alias if 'import X as Y'

@dataclass
class FunctionInfo:
    name: str
    qualified_name: str  # ClassName.method_name or module.function_name
    args: list[str]
    decorators: list[str]
    docstring: str
    start_line: int
    end_line: int
    is_async: bool = False
    is_method: bool = False

@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    methods: list[str]   # method names only
    docstring: str
    start_line: int
    end_line: int

@dataclass
class CallInfo:
    caller: str          # qualified name of calling function
    callee: str          # name being called (may be dotted)
    line: int

@dataclass
class FileSymbols:
    """All extracted symbols for one Python file."""
    file_path: str       # relative to repo root
    file_hash: str       # SHA-256 of file content
    imports: list[ImportInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    calls: list[CallInfo] = field(default_factory=list)
    module_docstring: str = ""
    parse_error: str = ""  # non-empty if Tree-sitter failed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FileSymbols":
        fs = cls(
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            module_docstring=data.get("module_docstring", ""),
            parse_error=data.get("parse_error", ""),
        )
        fs.imports = [ImportInfo(**i) for i in data.get("imports", [])]
        fs.functions = [FunctionInfo(**f) for f in data.get("functions", [])]
        fs.classes = [ClassInfo(**c) for c in data.get("classes", [])]
        fs.calls = [CallInfo(**c) for c in data.get("calls", [])]
        return fs

    @property
    def all_imported_modules(self) -> list[str]:
        """Top-level module names imported by this file."""
        mods = []
        for imp in self.imports:
            top = imp.module.split(".")[0]
            if top:
                mods.append(top)
        return list(set(mods))

    @property
    def summary_text(self) -> str:
        """
        Dense text summary for BM25 indexing.
        Includes: module docstring, function names, class names, import targets.
        """
        parts = []
        if self.module_docstring:
            parts.append(self.module_docstring)
        for fn in self.functions:
            parts.append(fn.name)
            if fn.docstring:
                parts.append(fn.docstring)
        for cls in self.classes:
            parts.append(cls.name)
            if cls.docstring:
                parts.append(cls.docstring)
            parts.extend(cls.methods)
        for imp in self.imports:
            parts.append(imp.module)
            parts.extend(imp.names)
        return " ".join(parts)


# ── Tree-sitter parser ────────────────────────────────────────────────────────

class PythonASTParser:
    """
    Parses Python files using Tree-sitter.

    Gracefully falls back to the stdlib `ast` module if Tree-sitter is
    unavailable (e.g. in minimal test environments).
    """

    def __init__(self):
        self._ts_available = False
        self._parser = None
        self._language = None
        self._try_init_treesitter()

    def _try_init_treesitter(self) -> None:
        """Attempt to load Tree-sitter; set flag if unavailable."""
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser
            self._language = Language(tspython.language())
            self._parser = Parser(self._language)
            self._ts_available = True
            logger.debug("Tree-sitter Python grammar loaded successfully")
        except Exception as e:
            logger.warning(
                "Tree-sitter not available, falling back to stdlib ast: %s", e
            )

    def parse_file(self, file_path: Path, repo_root: Path) -> FileSymbols:
        """
        Parse a single Python file and return its FileSymbols.

        Args:
            file_path: absolute path to the .py file
            repo_root: repo root for computing relative paths
        """
        try:
            source = file_path.read_bytes()
        except (OSError, PermissionError) as e:
            rel = str(file_path.relative_to(repo_root))
            return FileSymbols(
                file_path=rel,
                file_hash="",
                parse_error=f"Cannot read file: {e}",
            )

        file_hash = hashlib.sha256(source).hexdigest()
        rel_path = str(file_path.relative_to(repo_root))

        if self._ts_available:
            return self._parse_with_treesitter(source, file_hash, rel_path)
        else:
            return self._parse_with_stdlib_ast(source, file_hash, rel_path)

    def parse_repo(
        self,
        repo_root: Path,
        exclude_patterns: list[str] | None = None,
    ) -> Iterator[FileSymbols]:
        """
        Yield FileSymbols for every .py file in the repo.

        Args:
            repo_root: root directory of the repository
            exclude_patterns: glob patterns to exclude (e.g. ['test_*', 'setup.py'])
        """
        exclude_patterns = exclude_patterns or []
        py_files = [
            p for p in repo_root.rglob("*.py")
            if not any(part.startswith(".") for part in p.parts)
            and "__pycache__" not in str(p)
            and not any(p.match(pat) for pat in exclude_patterns)
        ]
        logger.info("Parsing %d Python files in %s", len(py_files), repo_root)
        for fp in py_files:
            yield self.parse_file(fp, repo_root)

    # ── Tree-sitter implementation ────────────────────────────────────────────

    def _parse_with_treesitter(
        self, source: bytes, file_hash: str, rel_path: str
    ) -> FileSymbols:
        """Full parse using Tree-sitter grammar."""
        tree = self._parser.parse(source)
        root = tree.root_node
        source_str = source.decode("utf-8", errors="replace")
        lines = source_str.splitlines()

        fs = FileSymbols(file_path=rel_path, file_hash=file_hash)

        # Track current class context for method qualification
        current_class: str | None = None

        def node_text(node) -> str:
            return source_str[node.start_byte:node.end_byte]

        def get_docstring(body_node) -> str:
            """Extract docstring from a function/class/module body."""
            if not body_node or body_node.named_child_count == 0:
                return ""
            first = body_node.named_children[0]
            if first.type == "expression_statement":
                inner = first.named_children[0] if first.named_children else None
                if inner and inner.type == "string":
                    raw = node_text(inner)
                    return raw.strip("\"'").strip()
            return ""

        # ── Module docstring ──────────────────────────────────────────────
        if root.named_child_count > 0:
            first = root.named_children[0]
            if first.type == "expression_statement" and first.named_children:
                inner = first.named_children[0]
                if inner.type == "string":
                    fs.module_docstring = node_text(inner).strip("\"'").strip()[:500]

        # ── Walk top-level nodes ──────────────────────────────────────────
        for node in root.named_children:
            if node.type in ("import_statement", "import_from_statement"):
                fs.imports.extend(self._extract_imports(node, node_text))

            elif node.type == "function_definition":
                fn = self._extract_function(node, node_text, get_docstring, None)
                fs.functions.append(fn)
                fs.calls.extend(self._extract_calls(node, node_text, fn.qualified_name))

            elif node.type == "class_definition":
                cls_info, methods, calls = self._extract_class(
                    node, node_text, get_docstring
                )
                fs.classes.append(cls_info)
                fs.functions.extend(methods)
                fs.calls.extend(calls)

            elif node.type == "decorated_definition":
                # decorated function or class
                inner = node.child_by_field_name("definition")
                if inner and inner.type == "function_definition":
                    fn = self._extract_function(
                        inner, node_text, get_docstring, None,
                        decorators=self._get_decorators(node, node_text)
                    )
                    fs.functions.append(fn)
                elif inner and inner.type == "class_definition":
                    cls_info, methods, calls = self._extract_class(
                        inner, node_text, get_docstring
                    )
                    fs.classes.append(cls_info)
                    fs.functions.extend(methods)
                    fs.calls.extend(calls)

        return fs

    def _extract_imports(self, node, node_text) -> list[ImportInfo]:
        imports = []
        if node.type == "import_statement":
            for name_node in node.named_children:
                if name_node.type in ("dotted_name", "aliased_import"):
                    if name_node.type == "aliased_import":
                        module = node_text(name_node.named_children[0])
                        alias = node_text(name_node.named_children[-1])
                    else:
                        module = node_text(name_node)
                        alias = ""
                    imports.append(ImportInfo(
                        module=module, names=[], is_from=False, alias=alias
                    ))
        elif node.type == "import_from_statement":
            # from X import Y, Z
            module_node = node.child_by_field_name("module_name")
            module = node_text(module_node) if module_node else ""
            names = []
            for child in node.named_children:
                if child.type in ("dotted_name", "identifier") and child != module_node:
                    names.append(node_text(child))
                elif child.type == "aliased_import":
                    names.append(node_text(child.named_children[0]))
                elif child.type == "wildcard_import":
                    names.append("*")
            imports.append(ImportInfo(module=module, names=names, is_from=True))
        return imports

    def _extract_function(
        self, node, node_text, get_docstring, class_name: str | None,
        decorators: list[str] | None = None
    ) -> FunctionInfo:
        name_node = node.child_by_field_name("name")
        name = node_text(name_node) if name_node else "<unknown>"
        qualified = f"{class_name}.{name}" if class_name else name

        # Parameters
        params_node = node.child_by_field_name("parameters")
        args = []
        if params_node:
            for param in params_node.named_children:
                if param.type == "identifier":
                    args.append(node_text(param))
                elif param.type in ("typed_parameter", "default_parameter",
                                    "typed_default_parameter"):
                    id_child = next(
                        (c for c in param.named_children if c.type == "identifier"), None
                    )
                    if id_child:
                        args.append(node_text(id_child))

        # Docstring
        body = node.child_by_field_name("body")
        docstring = get_docstring(body)[:300] if body else ""

        is_async = node.parent and node.parent.type == "decorated_definition" or \
                   any(c.type == "async" for c in node.children)

        return FunctionInfo(
            name=name,
            qualified_name=qualified,
            args=args,
            decorators=decorators or [],
            docstring=docstring,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            is_async="async_function_definition" in node.type or is_async,
            is_method=class_name is not None,
        )

    def _extract_class(
        self, node, node_text, get_docstring
    ) -> tuple[ClassInfo, list[FunctionInfo], list[CallInfo]]:
        name_node = node.child_by_field_name("name")
        class_name = node_text(name_node) if name_node else "<unknown>"

        # Base classes
        args_node = node.child_by_field_name("superclasses")
        bases = []
        if args_node:
            for child in args_node.named_children:
                if child.type in ("identifier", "dotted_name", "attribute"):
                    bases.append(node_text(child))

        body = node.child_by_field_name("body")
        docstring = get_docstring(body)[:300] if body else ""

        methods = []
        calls = []
        method_names = []

        if body:
            for child in body.named_children:
                if child.type in ("function_definition", "async_function_definition"):
                    fn = self._extract_function(child, node_text, get_docstring, class_name)
                    methods.append(fn)
                    method_names.append(fn.name)
                    calls.extend(self._extract_calls(child, node_text, fn.qualified_name))
                elif child.type == "decorated_definition":
                    inner = child.child_by_field_name("definition")
                    if inner and inner.type in ("function_definition", "async_function_definition"):
                        decs = self._get_decorators(child, node_text)
                        fn = self._extract_function(
                            inner, node_text, get_docstring, class_name, decs
                        )
                        methods.append(fn)
                        method_names.append(fn.name)
                        calls.extend(self._extract_calls(inner, node_text, fn.qualified_name))

        cls_info = ClassInfo(
            name=class_name,
            bases=bases,
            methods=method_names,
            docstring=docstring,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )
        return cls_info, methods, calls

    def _extract_calls(self, func_node, node_text, caller_name: str) -> list[CallInfo]:
        """Recursively find all call_expression nodes inside a function."""
        calls = []
        def walk(node):
            if node.type == "call":
                func_part = node.child_by_field_name("function")
                if func_part:
                    callee = node_text(func_part)
                    # Normalise to just the function name / dotted path
                    callee = callee.strip()
                    if len(callee) < 100:  # sanity limit
                        calls.append(CallInfo(
                            caller=caller_name,
                            callee=callee,
                            line=node.start_point[0] + 1,
                        ))
            for child in node.named_children:
                walk(child)
        walk(func_node)
        return calls

    def _get_decorators(self, decorated_node, node_text) -> list[str]:
        decorators = []
        for child in decorated_node.children:
            if child.type == "decorator":
                decorators.append(node_text(child).lstrip("@").strip())
        return decorators

    # ── stdlib ast fallback ───────────────────────────────────────────────────

    def _parse_with_stdlib_ast(
        self, source: bytes, file_hash: str, rel_path: str
    ) -> FileSymbols:
        """
        Fallback parser using stdlib `ast` module.
        Less detailed than Tree-sitter but always available.
        """
        import ast as stdlib_ast

        fs = FileSymbols(file_path=rel_path, file_hash=file_hash)
        source_str = source.decode("utf-8", errors="replace")

        try:
            tree = stdlib_ast.parse(source_str, filename=rel_path)
        except SyntaxError as e:
            fs.parse_error = str(e)
            return fs

        # Module docstring
        fs.module_docstring = stdlib_ast.get_docstring(tree) or ""

        for node in stdlib_ast.walk(tree):
            # Imports
            if isinstance(node, stdlib_ast.Import):
                for alias in node.names:
                    fs.imports.append(ImportInfo(
                        module=alias.name,
                        names=[],
                        is_from=False,
                        alias=alias.asname or "",
                    ))
            elif isinstance(node, stdlib_ast.ImportFrom):
                fs.imports.append(ImportInfo(
                    module=node.module or "",
                    names=[a.name for a in node.names],
                    is_from=True,
                ))

            # Functions
            elif isinstance(node, (stdlib_ast.FunctionDef, stdlib_ast.AsyncFunctionDef)):
                fs.functions.append(FunctionInfo(
                    name=node.name,
                    qualified_name=node.name,
                    args=[a.arg for a in node.args.args],
                    decorators=[stdlib_ast.unparse(d) for d in node.decorator_list],
                    docstring=(stdlib_ast.get_docstring(node) or "")[:300],
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    is_async=isinstance(node, stdlib_ast.AsyncFunctionDef),
                ))

            # Classes
            elif isinstance(node, stdlib_ast.ClassDef):
                methods = [
                    n.name for n in node.body
                    if isinstance(n, (stdlib_ast.FunctionDef, stdlib_ast.AsyncFunctionDef))
                ]
                fs.classes.append(ClassInfo(
                    name=node.name,
                    bases=[stdlib_ast.unparse(b) for b in node.bases],
                    methods=methods,
                    docstring=(stdlib_ast.get_docstring(node) or "")[:300],
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                ))

        return fs


# ── File hash helper (used by caching layer) ──────────────────────────────────

def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
