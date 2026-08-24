"""Deep content walker: unified traversal of page + Form XObject streams.

Production parsers see the SAME problem: text inside Form XObjects carries
the XObject's internal coordinate frame, which is meaningless until the
invoking `cm` transformation is applied. This module walks page and form
streams recursively, composing graphics states, so text fragments and
ruling-line segments land in ONE consistent device space.
"""

from __future__ import annotations

from typing import Any

from .table_structure_producer_v357_candidate import (
    RuledSegment,
    _apply,
    _mat_mult,
    _STROKE_OPS,
)


def _decode_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode("ascii", "replace")
    return str(name)


def _operand_str(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return v.decode("latin-1", "replace")
    return ""


def walk_page_content(
    page: Any,
    reader: Any,
) -> tuple[list[dict], list[RuledSegment]]:
    """Return (fragments, segments) in composed device space."""
    fragments: list[dict] = []
    segments: list[RuledSegment] = []
    visited: set[int] = set()
    # Keep referenced objects alive so their id() cannot be reused while
    # walking (prevents false cycle-detection hits).
    keepalive: list[Any] = []

    try:
        contents = page.get_contents()
        operations = list(getattr(contents, "operations", None) or ())
    except Exception:  # noqa: BLE001
        return fragments, segments

    resources = {}
    try:
        raw_resources = page.get("/Resources")
        if raw_resources is not None:
            resources = dict(raw_resources)
    except Exception:  # noqa: BLE001
        resources = {}

    _walk(
        operations,
        resources,
        reader,
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        fragments,
        segments,
        visited,
        depth=0,
        keepalive=keepalive,
    )
    return fragments, segments


def _resolve(resources: dict, category: str, name: str) -> Any:
    node = resources.get(category)
    if node is None:
        return None
    try:
        return dict(node).get("/" + name)
    except Exception:  # noqa: BLE001
        return None


MAX_XFORM_DEPTH = 32


def _walk(  # noqa: C901
    operations: list,
    resources: dict,
    reader: Any,
    ctm_in: list[float],
    fragments: list[dict],
    segments: list[RuledSegment],
    visited: set[int],
    depth: int = 0,
    keepalive: list[Any] | None = None,
) -> None:
    ctm = list(ctm_in)
    stack: list[list[float]] = []
    tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    size = 0.0
    in_text = False
    current: list[tuple[float, float]] = []
    pos = (0.0, 0.0)

    def flush_segments() -> None:
        nonlocal current
        if len(current) >= 2:
            for a, b in zip(current, current[1:]):
                segments.append(RuledSegment(a[0], a[1], b[0], b[1]))
        current = []

    for operands, operator in operations:
        name = (
            operator.decode("ascii", "replace")
            if isinstance(operator, bytes)
            else str(operator)
        )
        try:
            nums = [float(v) for v in operands]
        except (TypeError, ValueError):
            nums = []

        if name == "q":
            stack.append(ctm.copy())
        elif name == "Q":
            ctm = stack.pop() if stack else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif name == "cm" and len(nums) >= 6:
            ctm = _mat_mult(nums[:6], ctm)
        elif name == "BT":
            in_text = True
            tm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif name == "ET":
            in_text = False
        elif name == "Tf" and len(operands) >= 2:
            try:
                size = float(operands[1])
            except (TypeError, ValueError):
                pass
        elif name == "Tm" and len(nums) >= 6:
            tm = nums[:6]
            pos = _apply(ctm, tm[4], tm[5])
        elif name == "Td" and len(nums) >= 2:
            shifted = [
                tm[0], tm[1], tm[2], tm[3],
                tm[4] + nums[0] * tm[0] + nums[1] * tm[2],
                tm[5] + nums[0] * tm[1] + nums[1] * tm[3],
            ]
            tm = shifted
            pos = _apply(ctm, tm[4], tm[5])
        elif name == "TL" and len(nums) >= 1:
            pass
        elif name == "T*":
            pass
        elif name in {"Tj", "'", '"'} and in_text:
            text = "".join(_operand_str(v) for v in operands if not isinstance(v, list))
            if text.strip():
                px, py = _apply(ctm, tm[4], tm[5])
                fragments.append(
                    {"text": text.strip(), "x": px, "y": py, "size": size},
                )
        elif name == "TJ" and in_text:
            pieces: list[str] = []
            for operand in operands:
                if isinstance(operand, (list, tuple)):
                    for item in operand:
                        s = _operand_str(item)
                        if s:
                            pieces.append(s)
                else:
                    s = _operand_str(operand)
                    if s:
                        pieces.append(s)
            text = "".join(pieces)
            if text.strip():
                px, py = _apply(ctm, tm[4], tm[5])
                fragments.append(
                    {"text": text.strip(), "x": px, "y": py, "size": size},
                )
        elif name == "Do":
            flush_segments()
            if depth >= MAX_XFORM_DEPTH:
                continue
            xname = _decode_name(operands[0]) if operands else ""
            xobj = _resolve(resources, "XObject", xname)
            if xobj is None:
                continue
            try:
                subtype = str(xobj.get("/Subtype", ""))
            except Exception:  # noqa: BLE001
                subtype = ""
            if subtype != "/Form":
                continue
            marker = id(xobj)
            if marker in visited:
                continue
            visited.add(marker)
            if keepalive is not None:
                keepalive.append(xobj)
            try:
                child_resources = {}
                raw_res = xobj.get("/Resources")
                if raw_res is not None:
                    child_resources = dict(raw_res)
            except Exception:  # noqa: BLE001
                child_resources = {}
            from pypdf.generic import ContentStream

            try:
                stream = ContentStream(xobj, reader)
                child_ops = list(getattr(stream, "operations", None) or [])
            except Exception:  # noqa: BLE001
                child_ops = []
            _walk(
                child_ops,
                child_resources or resources,
                reader,
                ctm,
                fragments,
                segments,
                visited,
                depth=depth + 1,
            )
        elif name == "m" and len(nums) >= 2:
            flush_segments()
            pos = _apply(ctm, nums[0], nums[1])
            current = [pos]
        elif name == "l" and len(nums) >= 2:
            pos = _apply(ctm, nums[0], nums[1])
            current.append(pos)
        elif name == "c" and len(nums) >= 6:
            pos = _apply(ctm, nums[4], nums[5])
            current.append(pos)
        elif name == "re" and len(nums) >= 4:
            x, y, w, h = nums[:4]
            corners = [
                _apply(ctm, x, y),
                _apply(ctm, x + w, y),
                _apply(ctm, x + w, y + h),
                _apply(ctm, x, y + h),
                _apply(ctm, x, y),
            ]
            for a, b in zip(corners, corners[1:]):
                segments.append(RuledSegment(a[0], a[1], b[0], b[1]))
            pos = corners[-1]
        elif name in _STROKE_OPS:
            flush_segments()
    flush_segments()
