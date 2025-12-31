#!/usr/bin/env python3
import time
import argparse
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def path_to_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def uri_to_path(uri: str) -> str:
    if uri.startswith("file://"):
        return Path(uri[len("file://"):]).as_posix()
    return uri


def pipe_stderr(proc: subprocess.Popen):
    if proc.stderr is None:
        return
    for line in proc.stderr:
        sys.stderr.buffer.write(line)
        sys.stderr.flush()


class LSPClient:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._next_id = 1
        self._lock = threading.Lock()

    def _send(self, msg: dict):
        data = json.dumps(msg, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        assert self.proc.stdin is not None
        self.proc.stdin.write(header + data)
        self.proc.stdin.flush()

    def _read_message(self) -> dict:
        assert self.proc.stdout is not None
        headers = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("clangd stdout closed")
            line = line.decode("ascii", errors="replace").strip()
            if line == "":
                break
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

        length = int(headers["content-length"])
        body = self.proc.stdout.read(length)
        return json.loads(body.decode("utf-8"))

    def request(self, method: str, params: dict):
        with self._lock:
            rid = self._next_id
            self._next_id += 1

        self._send({"jsonrpc": "2.0", "id": rid,
                   "method": method, "params": params})

        while True:
            msg = self._read_message()
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result")

    def notify(self, method: str, params: dict):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def shutdown(self):
        try:
            self.request("shutdown", {})
        except Exception:
            pass
        try:
            self.notify("exit", {})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


def fmt_loc(rng):
    s = rng["start"]
    return f'{s["line"]+1}:{s["character"]+1}'


def fmt_item(it):
    name = it.get("name")
    detail = it.get("detail")
    u = it.get("uri")
    loc = it.get("range", {})
    loc_s = ""
    if isinstance(loc, dict) and "start" in loc:
        loc_s = fmt_loc(loc)
    return f"{name}{(' ' + detail) if detail else ''} ({uri_to_path(u)}){(' @ ' + loc_s) if loc_s else ''}"


def find_identifier_pos_in_text(text: str, ident: str, occurrence: int = 1) -> Tuple[int, int]:
    pattern = re.compile(rf"\b{re.escape(ident)}\b")
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"identifier not found as a whole word: {ident}")
    if occurrence < 1 or occurrence > len(matches):
        raise ValueError(
            f"occurrence {occurrence} out of range (found {len(matches)} matches)")
    m = matches[occurrence - 1]
    idx = m.start()

    before = text[:idx]
    line = before.count("\n")
    col = idx - (before.rfind("\n") + 1 if "\n" in before else 0)
    return line, col + 1


@dataclass
class Loc:
    uri: str
    line: int
    character: int


def loc_from_lsp_location(loc: Dict[str, Any]) -> Loc:
    # Location = { uri, range: {start,end} }.
    # locationLink = { targetUri, targetRange, targetSelectionRange, originSelectionRange? }
    if "uri" in loc and "range" in loc:
        s = loc["range"]["start"]
        return Loc(loc["uri"], s["line"], s["character"])
    if "targetUri" in loc and "targetSelectionRange" in loc:
        s = loc["targetSelectionRange"]["start"]
        return Loc(loc["targetUri"], s["line"], s["character"])
    if "targetUri" in loc and "targetRange" in loc:
        s = loc["targetRange"]["start"]
        return Loc(loc["targetUri"], s["line"], s["character"])
    raise ValueError(f"unrecognized location shape: {loc.keys()}")


def did_open(client: LSPClient, uri: str, language_id: str, text: str, version: int = 1):
    client.notify("textDocument/didOpen", {
        "textDocument": {"uri": uri, "languageId": language_id, "version": version, "text": text}
    })


def request_definition(client: LSPClient, loc: Loc) -> List[Loc]:
    res = client.request("textDocument/definition", {
        "textDocument": {"uri": loc.uri},
        "position": {"line": loc.line, "character": loc.character},
    })
    if res is None:
        return []
    if isinstance(res, dict):
        return [loc_from_lsp_location(res)]
    if isinstance(res, list):
        return [loc_from_lsp_location(x) for x in res]
    return []


def request_workspace_symbol(client, query, tries=50, sleep_s=0.1):
    for _ in range(tries):
        res = client.request("workspace/symbol", {"query": query}) or []
        if isinstance(res, list) and res:
            return res
        time.sleep(sleep_s)
    return []


def symbol_name(sym: Dict[str, Any]) -> str:
    return sym.get("name") or ""


def symbol_kind(sym: Dict[str, Any]) -> Optional[int]:
    # LSP SymbolKind enum: 12=Function, 6=Method, 9=Constructor, etc.
    return sym.get("kind")


def symbol_location(sym: Dict[str, Any]) -> Optional[Loc]:
    # SymbolInformation: { name, kind, location: {uri,range}, containerName? }
    if "location" in sym:
        try:
            return loc_from_lsp_location(sym["location"])
        except Exception:
            return None
    # WorkspaceSymbol: { name, kind, location: {uri/range} OR {range,uri} OR locationLink }
    if "location" in sym:
        try:
            return loc_from_lsp_location(sym["location"])
        except Exception:
            return None
    return None


def is_header(path: Path) -> bool:
    return path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx", ".inc"}


def is_source(path: Path) -> bool:
    return path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".m", ".mm"}


def score_candidate(project_root: Path, loc: Loc) -> Tuple[int, int, int]:
    """
    Higher is better.
    Tuple ordering: (in_project, is_source, not_header)
    """
    p = Path(uri_to_path(loc.uri))
    in_proj = 1 if str(p.resolve()).startswith(
        str(project_root.resolve())) else 0
    src = 1 if is_source(p) else 0
    not_hdr = 1 if not is_header(p) else 0
    return (in_proj, src, not_hdr)


def prepare_call_hierarchy_at(client: LSPClient, loc: Loc) -> List[Dict[str, Any]]:
    return client.request("textDocument/prepareCallHierarchy", {
        "textDocument": {"uri": loc.uri},
        "position": {"line": loc.line, "character": loc.character},
    }) or []


def run_call_hierarchy(client: LSPClient, item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    incoming = client.request(
        "callHierarchy/incomingCalls", {"item": item}) or []
    try:
        outgoing = client.request(
            "callHierarchy/outgoingCalls", {"item": item}) or []
    except RuntimeError as e:
        err = e.args[0] if e.args else None
        if isinstance(err, dict) and err.get("code") == -32601:
            outgoing = []
        else:
            raise
    return incoming, outgoing


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Call hierarchy CLI using clangd (LSP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--compile-commands-dir", required=True,
                   help="Directory containing compile_commands.json")
    p.add_argument("--project-root", default=None,
                   help="Project root for workspace/symbol + candidate scoring. Defaults to --compile-commands-dir.")
    p.add_argument("--function", required=True,
                   help="Function/symbol name to query (identifier).")
    p.add_argument("--file", default=None,
                   help="Optional: file to force v1 behavior (search identifier within this file).")
    p.add_argument("--occurrence", type=int, default=1,
                   help="v1 only: if identifier appears multiple times in --file, pick Nth match.")
    p.add_argument("--language", default="c",
                   help="languageId for didOpen when opening text (c/cpp/etc).")

    p.add_argument("--pick", default="best",
                   help="How to choose when multiple defs exist: best | list | N (1-based index)")
    p.add_argument("--json", action="store_true",
                   help="Output machine-readable JSON.")
    p.add_argument("--check-hover", action="store_true",
                   help="Do a hover request before call hierarchy (sanity check).")

    p.add_argument("--clangd", default="clangd",
                   help="clangd executable path.")
    p.add_argument("--clangd-log", default=None,
                   help="clangd --log level (error, info, verbose).")
    p.add_argument("--stream-stderr", action="store_true",
                   help="Stream clangd stderr to this terminal.")
    p.add_argument("clangd_args", nargs=argparse.REMAINDER,
                   help="Extra args after -- are passed to clangd")
    ns = p.parse_args(argv)
    if ns.clangd_args and ns.clangd_args[0] == "--":
        ns.clangd_args = ns.clangd_args[1:]
    return ns


def open_anchor_from_compdb(client, cc_dir: Path, language_id: str):
    db = json.loads((cc_dir / "compile_commands.json").read_text())
    for e in db:
        f = e.get("file")
        if not f:
            continue

        p = Path(f)
        if not p.is_absolute():
            p = Path(e.get("directory", cc_dir)) / p

        if p.exists():
            uri = path_to_uri(str(p))
            text = p.read_text(encoding="utf-8", errors="replace")
            client.notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            })
            return True
    return False


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])

    cc_dir = Path(args.compile_commands_dir).resolve()
    if not cc_dir.is_dir():
        print(
            f"error: --compile-commands-dir is not a directory: {cc_dir}", file=sys.stderr)
        return 2
    if not (cc_dir / "compile_commands.json").exists():
        print(
            f"error: compile_commands.json not found in: {cc_dir}", file=sys.stderr)
        return 2

    project_root = Path(args.project_root).resolve(
    ) if args.project_root else cc_dir

    clangd_cmd = [args.clangd,
                  f"--compile-commands-dir={cc_dir}", "--background-index"]
    if args.clangd_log:
        clangd_cmd.append(f"--log={args.clangd_log}")
    else:
        clangd_cmd.append("--log=error")
    if args.clangd_args:
        clangd_cmd.extend(args.clangd_args)

    client = LSPClient(clangd_cmd)
    if args.stream_stderr:
        threading.Thread(target=pipe_stderr, args=(
            client.proc,), daemon=True).start()

    try:
        _ = client.request("initialize", {
            "processId": os.getpid(),
            "rootUri": path_to_uri(str(project_root)),
            "workspaceFolders": [{
                "uri": path_to_uri(str(project_root)),
                "name": project_root.name,
            }],
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "callHierarchy": {}
                },
                "workspace": {
                    "symbol": {}
                }
            }
        })
        client.notify("initialized", {})
        open_anchor_from_compdb(client, cc_dir, args.language)

        # --- v1 path: user provided --file (keep your proven behavior) ---
        if args.file:
            file_path = Path(args.file).resolve()
            if not file_path.exists():
                print(f"error: file not found: {file_path}", file=sys.stderr)
                return 2

            uri = path_to_uri(str(file_path))
            text = file_path.read_text(encoding="utf-8", errors="replace")
            did_open(client, uri, args.language, text)

            line, ch = find_identifier_pos_in_text(
                text, args.function, args.occurrence)
            target_loc = Loc(uri, line, ch)

        # --- v2 path: find symbol globally ---
        else:
            syms = request_workspace_symbol(client, args.function)

            # Filter: exact name match + function-ish kinds if present.
            # SymbolKind: 12 Function, 6 Method, 9 Constructor
            allowed_kinds = {12, 6, 9}
            candidates: List[Tuple[Dict[str, Any], Loc]] = []

            def name_matches(got, want):
                return got == want or got.endswith("::" + want)

            print(syms)

            for s in syms:
                print(s)
                if not name_matches(symbol_name(s), args.function):
                    continue
                k = symbol_kind(s)
                if k is not None and k not in allowed_kinds:
                    continue
                loc = symbol_location(s)
                if loc:
                    candidates.append((s, loc))

            if not candidates:
                print(
                    f"error: workspace/symbol found no exact matches for '{args.function}'", file=sys.stderr)
                return 1

            # Resolve each candidate to definition(s)
            def_locs: List[Loc] = []
            for (_sym, loc) in candidates:
                # clangd may work without didOpen, but opening tends to improve reliability.
                p = Path(uri_to_path(loc.uri))
                if p.exists():
                    try:
                        txt = p.read_text(encoding="utf-8", errors="replace")
                        did_open(client, loc.uri, args.language, txt)
                    except Exception:
                        pass
                defs = request_definition(client, loc)
                def_locs.extend(defs)

            if not def_locs:
                # fallback: use candidate location(s) directly
                def_locs = [loc for (_sym, loc) in candidates]

            # Deduplicate by (uri,line,char)
            uniq = {}
            for dl in def_locs:
                uniq[(dl.uri, dl.line, dl.character)] = dl
            def_locs = list(uniq.values())

            # Order by heuristic
            def_locs.sort(key=lambda l: score_candidate(
                project_root, l), reverse=True)

            if args.pick == "list":
                for i, l in enumerate(def_locs, 1):
                    print(
                        f"{i}. {uri_to_path(l.uri)} @ {l.line+1}:{l.character+1}  score={score_candidate(project_root, l)}")
                return 0

            chosen: Optional[Loc] = None
            if args.pick == "best":
                chosen = def_locs[0]
            else:
                # numeric pick
                try:
                    n = int(args.pick)
                    if not (1 <= n <= len(def_locs)):
                        raise ValueError()
                    chosen = def_locs[n - 1]
                except Exception:
                    print(
                        "error: --pick must be 'best', 'list', or an integer index (1-based)", file=sys.stderr)
                    return 2

            # Open chosen doc (helps clangd)
            chosen_path = Path(uri_to_path(chosen.uri))
            if chosen_path.exists():
                text = chosen_path.read_text(
                    encoding="utf-8", errors="replace")
                did_open(client, chosen.uri, args.language, text)

            target_loc = chosen

        # Optional hover sanity check
        if args.check_hover:
            hover = client.request("textDocument/hover", {
                "textDocument": {"uri": target_loc.uri},
                "position": {"line": target_loc.line, "character": target_loc.character}
            })
            if hover is None:
                raise RuntimeError(
                    "hover returned null (wrong position / flags / file?)")

        items = prepare_call_hierarchy_at(client, target_loc)
        if not items:
            raise RuntimeError("No call hierarchy item at resolved location")

        item = items[0]
        incoming, outgoing = run_call_hierarchy(client, item)

        if args.json:
            out = {
                "resolved": {"uri": target_loc.uri, "line": target_loc.line, "character": target_loc.character},
                "target": item,
                "incoming": incoming,
                "outgoing": outgoing,
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

        print("TARGET:")
        print("  " + fmt_item(item))

        print("\nCALLERS:")
        for call in incoming:
            frm = call["from"]
            locs = ", ".join(fmt_loc(r) for r in call.get("fromRanges", []))
            print(f"  {fmt_item(frm)}" +
                  (f"  callsites: {locs}" if locs else ""))

        print("\nCALLEES:")
        for call in outgoing:
            to = call["to"]
            locs = ", ".join(fmt_loc(r) for r in call.get("fromRanges", []))
            print(f"  {fmt_item(to)}" +
                  (f"  callsites: {locs}" if locs else ""))

        return 0

    except Exception as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1
    finally:
        client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
