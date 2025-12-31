#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple


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
        data = json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
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

        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})

        while True:
            msg = self._read_message()
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result")
            # ignore other messages for this CLI (notifications, etc.)

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


def find_identifier_pos(text: str, ident: str, occurrence: int = 1) -> Tuple[int, int]:
    """
    Find ident as a word boundary match and return (line, character).
    Character counting is UTF-16 in LSP, but for ASCII identifiers this is OK.
    """
    pattern = re.compile(rf"\b{re.escape(ident)}\b")
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"identifier not found as a whole word: {ident}")
    if occurrence < 1 or occurrence > len(matches):
        raise ValueError(f"occurrence {occurrence} out of range (found {len(matches)} matches)")

    m = matches[occurrence - 1]
    idx = m.start()

    before = text[:idx]
    line = before.count("\n")
    col = idx - (before.rfind("\n") + 1 if "\n" in before else 0)

    # Aim inside the identifier token (not at boundary).
    return line, col + 1


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


def run(args) -> int:
    # Resolve compile_commands.json directory
    cc_dir = Path(args.compile_commands_dir).resolve()
    if not cc_dir.is_dir():
        print(f"error: --compile-commands-dir is not a directory: {cc_dir}", file=sys.stderr)
        return 2
    if not (cc_dir / "compile_commands.json").exists():
        print(f"error: compile_commands.json not found in: {cc_dir}", file=sys.stderr)
        return 2

    # Resolve file
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"error: file not found: {file_path}", file=sys.stderr)
        return 2

    clangd_cmd = [args.clangd, f"--compile-commands-dir={cc_dir}"]
    # keep logs quiet unless requested
    if args.clangd_log:
        clangd_cmd.append(f"--log={args.clangd_log}")
    else:
        clangd_cmd.append("--log=error")

    # Allow extra clangd args after --
    if args.clangd_args:
        clangd_cmd.extend(args.clangd_args)

    client = LSPClient(clangd_cmd)
    if args.stream_stderr:
        threading.Thread(target=pipe_stderr, args=(client.proc,), daemon=True).start()

    root_uri = path_to_uri(str(cc_dir))

    try:
        _ = client.request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "callHierarchy": {}
                }
            }
        })
        client.notify("initialized", {})

        uri = path_to_uri(str(file_path))
        text = file_path.read_text(encoding="utf-8", errors="replace")

        client.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": args.language,
                "version": 1,
                "text": text
            }
        })

        line, ch = find_identifier_pos(text, args.function, args.occurrence)

        if args.check_hover:
            hover = client.request("textDocument/hover", {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": ch}
            })
            if hover is None:
                raise RuntimeError("hover returned null (wrong position / flags / file?)")

        items = client.request("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": ch}
        })
        if not items:
            raise RuntimeError("No call hierarchy item at position (are you on the function name?)")

        item = items[0]
        incoming = client.request("callHierarchy/incomingCalls", {"item": item}) or []

        outgoing = []
        try:
            outgoing = client.request("callHierarchy/outgoingCalls", {"item": item}) or []
        except RuntimeError as e:
            err = e.args[0] if e.args else None
            if isinstance(err, dict) and err.get("code") == -32601:
                outgoing = []
            else:
                raise

        if args.json:
            out = {"target": item, "incoming": incoming, "outgoing": outgoing}
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

        print("TARGET:")
        print("  " + fmt_item(item))

        print("\nCALLERS:")
        for call in incoming:
            frm = call["from"]
            locs = ", ".join(fmt_loc(r) for r in call.get("fromRanges", []))
            print(f"  {fmt_item(frm)}" + (f"  callsites: {locs}" if locs else ""))

        print("\nCALLEES:")
        for call in outgoing:
            to = call["to"]
            locs = ", ".join(fmt_loc(r) for r in call.get("fromRanges", []))
            print(f"  {fmt_item(to)}" + (f"  callsites: {locs}" if locs else ""))

        return 0

    except Exception as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1
    finally:
        client.shutdown()


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Call hierarchy CLI using clangd (LSP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--compile-commands-dir", required=True,
                   help="Directory containing compile_commands.json")
    p.add_argument("--file", required=True,
                   help="Source file to open (path).")
    p.add_argument("--function", required=True,
                   help="Function/symbol name to query (identifier).")
    p.add_argument("--occurrence", type=int, default=1,
                   help="If identifier appears multiple times, pick Nth match in the file.")
    p.add_argument("--language", default="c",
                   help="languageId for didOpen (c, cpp, objective-c, etc.)")
    p.add_argument("--clangd", default="clangd",
                   help="clangd executable path.")
    p.add_argument("--clangd-log", default=None,
                   help="clangd --log level (error, info, verbose).")
    p.add_argument("--stream-stderr", action="store_true",
                   help="Stream clangd stderr to this terminal.")
    p.add_argument("--check-hover", action="store_true",
                   help="Do a hover request before call hierarchy (sanity check).")
    p.add_argument("--json", action="store_true",
                   help="Output machine-readable JSON.")
    # Everything after -- is passed to clangd
    p.add_argument("clangd_args", nargs=argparse.REMAINDER,
                   help="Extra args after -- are passed to clangd")
    ns = p.parse_args(argv)
    # Strip leading "--" from remainder if present
    if ns.clangd_args and ns.clangd_args[0] == "--":
        ns.clangd_args = ns.clangd_args[1:]
    return ns


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

