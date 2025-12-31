import json
import os
import subprocess
import threading
import sys

from pathlib import Path


def path_to_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


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
        # Read headers
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
        # Simple blocking wait for matching id (good enough for POC)
        while True:
            msg = self._read_message()
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]
            # else: ignore notifications/other responses for this POC

    def notify(self, method: str, params: dict):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})


def find_pos(text: str, needle: str, occurrence: int = 1) -> tuple[int, int]:
    # Returns (line, character) in Python codepoints (OK for ASCII POC).
    start = 0
    for _ in range(occurrence):
        idx = text.find(needle, start)
        if idx == -1:
            raise ValueError(f"'{needle}' not found")
        start = idx + len(needle)

    before = text[:idx]
    line = before.count("\n")
    col = idx - (before.rfind("\n") + 1 if "\n" in before else 0)
    return line, col + max(0, len(needle)//2)  # aim inside the identifier

def path_to_uri(path: str) -> str:
    return Path(path).resolve().as_uri()

def uri_to_path(uri: str) -> str:
    # Works for file:// URIs clangd returns
    if uri.startswith("file://"):
        return Path(uri[len("file://"):]).as_posix()
    return uri

def pipe_stderr(proc):
    if proc.stderr is None:
        return
    for line in proc.stderr:
        sys.stderr.buffer.write(line)
        sys.stderr.flush()

def find_pos(text: str, needle: str, occurrence: int = 1) -> tuple[int, int]:
    """
    Returns (line, character) for an ASCII-safe POC by searching for `needle`.
    LSP character is UTF-16 code units; this is correct for ASCII.
    """
    start = 0
    idx = -1
    for _ in range(occurrence):
        idx = text.find(needle, start)
        if idx == -1:
            raise ValueError(f"'{needle}' not found")
        start = idx + len(needle)

    before = text[:idx]
    line = before.count("\n")
    col = idx - (before.rfind("\n") + 1 if "\n" in before else 0)
    # Aim inside the identifier, not at boundary.
    return line, col + max(1, len(needle) // 2)

def main():
    # Start clangd. Ensure compile_commands.json exists in cwd
    # or pass --compile-commands-dir=... for better results.
    client = LSPClient(["clangd", "--log=error"])

    # Optional: stream clangd logs to your terminal
    threading.Thread(target=pipe_stderr, args=(client.proc,), daemon=True).start()

    root_uri = path_to_uri(os.getcwd())
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

    file_path = "src/storage/page_buffer.c"
    uri = path_to_uri(file_path)

    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    client.notify("textDocument/didOpen", {
        "textDocument": {
            "uri": uri,
            "languageId": "c",
            "version": 1,
            "text": text
        }
    })

    # Pick the symbol to query
    needle = "pgbuf_fix_debug"  # change this
    line, ch = find_pos(text, needle)

    # (Optional) verify hover is non-null
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

    try:
        outgoing = client.request("callHierarchy/outgoingCalls", {"item": item}) or []
    except RuntimeError as e:
        err = getattr(e, "args", [None])[0]
        # your code raises RuntimeError(msg["error"]), so err is likely a dict
        if isinstance(err, dict) and err.get("code") == -32601:
            outgoing = []
        else:
            raise


    def fmt_loc(rng):
        s = rng["start"]
        return f'{s["line"]+1}:{s["character"]+1}'

    def fmt_item(it):
        name = it.get("name")
        detail = it.get("detail")
        u = it.get("uri")
        loc = it.get("range", {})
        loc_s = ""
        if "start" in loc:
            loc_s = fmt_loc(loc)
        return f"{name}{(' ' + detail) if detail else ''} ({uri_to_path(u)}){(' @ ' + loc_s) if loc_s else ''}"

    print("HOVER (raw):")
    print(json.dumps(hover, ensure_ascii=False, indent=2))

    print("\nTARGET:")
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

if __name__ == "__main__":
    main()
