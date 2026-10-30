# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from ._errors import InvalidSegmentationMaskError
from ._mask_codec import decode_mask_to_raster
from ._types import (
    FrameReference,
    ImageSegmentationResult,
    ImageSegmentationSnapshot,
    ParserFinish,
    ResponseFormat,
    ResponseStreamOutcome,
    SegmentationBoxRecord,
    SegmentationDiagnostic,
    SegmentationMask,
    SegmentationMaskBounds,
    SegmentationMaskIdentity,
    SegmentationMaskRecord,
    SegmentationMedia,
    SegmentationRecord,
    SegmentationResult,
    SegmentationSnapshot,
    SegmentationTextRecord,
    VideoSegmentationResult,
    VideoSegmentationSnapshot,
)

_MAXIMUM_SAFE_INTEGER = (1 << 53) - 1
_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_JS_WHITESPACE_CLASS = (
    r"\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_JS_DOT_PATTERN = r"[^\n\r\u2028\u2029]"
_API_HEADER_PATTERN = re.compile(rf"^<([0-9]+)f>({_JS_DOT_PATTERN}*)$")
_API_RECORD_PATTERN = re.compile(
    r"^(?:,)?([0-9]+)"
    r"<\|box;x1=(-?[0-9]+);y1=(-?[0-9]+);x2=(-?[0-9]+);"
    r"y2=(-?[0-9]+);w=([0-9]+);h=([0-9]+)\|>"
    r"<\|mask;x=0;y=0;data=([0-9]+),([0-9]+),([!~][^|]+)\|>"
)


def _trim(value: str) -> str:
    return value.strip(_JS_WHITESPACE)


def _integer_text(value: str) -> int | None:
    """Parse a decimal integer token the way JavaScript's ``Number`` observes it.

    Callers only pass regex captures of ``-?[0-9]+``; values beyond the
    JavaScript safe-integer range are rejected so both parsers agree.
    """
    parsed = int(value)
    return parsed if abs(parsed) <= _MAXIMUM_SAFE_INTEGER else None


class _SegmentationParser:
    __slots__ = (
        "_buffer_parts",
        "_diagnostics",
        "_line",
        "_media",
        "_raw_output_parts",
        "_records",
        "_revision",
        "_revisions",
    )

    def __init__(self, media: SegmentationMedia) -> None:
        self._media = media
        self._records: list[SegmentationRecord] = []
        self._diagnostics: list[SegmentationDiagnostic] = []
        self._revisions: dict[SegmentationMaskIdentity, int] = {}
        self._raw_output_parts: list[str] = []
        self._buffer_parts: list[str] = []
        self._line = 0
        self._revision = 0

    def push(
        self, chunk: str, *, emit: bool = True
    ) -> tuple[SegmentationSnapshot, ...]:
        if not isinstance(chunk, str):
            raise TypeError("Parser chunks must be strings.")
        self._raw_output_parts.append(chunk)

        changed = False
        segments = chunk.split("\n")
        for segment in segments[:-1]:
            self._append_buffer(segment)
            raw = "".join(self._buffer_parts)
            if raw.endswith("\r"):
                raw = raw[:-1]
            self._clear_buffer()
            prior = (len(self._records), len(self._diagnostics))
            self._accept_line(raw)
            changed = changed or prior != (
                len(self._records),
                len(self._diagnostics),
            )
        self._append_buffer(segments[-1])

        if changed:
            self._revision += 1
        return (self._snapshot(),) if changed and emit is not False else ()

    def finish(
        self, outcome: ResponseStreamOutcome
    ) -> ParserFinish[SegmentationSnapshot, SegmentationResult]:
        events: tuple[SegmentationSnapshot, ...] = ()
        if self._buffer_parts:
            raw = "".join(self._buffer_parts)
            if raw.endswith("\r"):
                raw = raw[:-1]
            self._clear_buffer()
            prior = (len(self._records), len(self._diagnostics))
            self._accept_line(raw)
            if prior != (len(self._records), len(self._diagnostics)):
                self._revision += 1
                events = (self._snapshot(),)
        return ParserFinish(events=events, result=self._result(outcome))

    def _append_buffer(self, value: str) -> None:
        if value:
            self._buffer_parts.append(value)

    def _clear_buffer(self) -> None:
        self._buffer_parts.clear()

    def _accept_line(self, raw: str) -> bool:
        self._line += 1
        line = _trim(raw)
        if not line:
            return False

        if line.startswith("<"):
            accepted = self._accept_api_line(line, raw)
            if accepted is not None:
                return accepted

        self._add_record(SegmentationTextRecord(order=len(self._records), text=raw))
        return True

    def _accept_api_line(self, line: str, raw: str) -> bool | None:
        header = _API_HEADER_PATTERN.fullmatch(line)
        if header is None:
            return None
        frame_index = _integer_text(header.group(1))
        if frame_index is None or frame_index < 0:
            self._diagnose(
                "invalid_frame",
                "Frame references must be non-negative safe integers.",
                raw,
            )
            return True
        if self._media == "image" and frame_index != 0:
            self._diagnose(
                "unexpected_frame",
                "Image segmentation records require frame zero.",
                raw,
            )
            return True

        frame = FrameReference(frame_index) if self._media == "video" else None
        remainder = header.group(2)
        if not remainder:
            self._diagnose(
                "malformed_record",
                "Malformed SAM API object record.",
                raw,
            )
            return True
        accepted = False
        while remainder:
            match = _API_RECORD_PATTERN.match(remainder)
            if match is None:
                self._diagnose(
                    "malformed_record",
                    "Malformed SAM API object record.",
                    raw,
                )
                return True
            object_id = match.group(1)
            coordinate_values = tuple(
                _integer_text(value) for value in match.group(2, 3, 4, 5, 6, 7)
            )
            if any(value is None for value in coordinate_values):
                self._diagnose(
                    "invalid_box", "SAM API box coordinates are invalid.", raw
                )
                return True
            (
                left,
                top,
                inclusive_right,
                inclusive_bottom,
                source_width,
                source_height,
            ) = cast(tuple[int, int, int, int, int, int], coordinate_values)
            if (
                source_width <= 0
                or source_height <= 0
                or left < 0
                or top < 0
                or inclusive_right < left
                or inclusive_bottom < top
                or inclusive_right >= source_width
                or inclusive_bottom >= source_height
            ):
                self._diagnose(
                    "invalid_box", "SAM API box coordinates are invalid.", raw
                )
                return True
            bounds = SegmentationMaskBounds(
                left=left,
                top=top,
                right=inclusive_right + 1,
                bottom=inclusive_bottom + 1,
            )
            self._add_record(
                SegmentationBoxRecord(
                    order=len(self._records),
                    object_id=object_id,
                    frame=frame,
                    left=left,
                    top=top,
                    right=inclusive_right + 1,
                    bottom=inclusive_bottom + 1,
                )
            )
            payload = match.group(10)
            self._accept_mask(
                object_id,
                frame,
                SegmentationMask(
                    encoding="lossless" if payload.startswith("~") else "one_bit",
                    payload=payload,
                    width=int(match.group(9)),
                    height=int(match.group(8)),
                ),
                raw,
                bounds,
            )
            accepted = True
            remainder = remainder[match.end() :]
        return accepted

    def _accept_mask(
        self,
        object_id: str,
        frame: FrameReference | None,
        mask: SegmentationMask,
        raw: str,
        bounds: SegmentationMaskBounds,
    ) -> None:
        area = mask.width * mask.height
        if mask.width <= 0 or mask.height <= 0 or area > _MAXIMUM_SAFE_INTEGER:
            self._diagnose(
                "invalid_mask_size", "Mask dimensions must be positive.", raw
            )
            return
        try:
            decode_mask_to_raster(mask)
        except InvalidSegmentationMaskError as error:
            self._diagnose("invalid_mask_payload", str(error), raw)
            return

        identity = SegmentationMaskIdentity(
            media=self._media,
            frame_index=None if frame is None else frame.frame_index,
            object_id=object_id,
        )
        revision = self._revisions.get(identity, 0) + 1
        self._revisions[identity] = revision
        self._add_record(
            SegmentationMaskRecord(
                order=len(self._records),
                object_id=object_id,
                frame=frame,
                identity=identity,
                revision=revision,
                mask=mask,
                bounds=bounds,
            )
        )

    def _add_record(self, record: SegmentationRecord) -> None:
        self._records.append(record)

    def _diagnose(self, code: str, message: str, raw: str) -> None:
        self._diagnostics.append(
            SegmentationDiagnostic(
                severity="error",
                code=code,
                message=message,
                line=self._line,
                raw=raw,
            )
        )

    def _snapshot(self) -> SegmentationSnapshot:
        records = tuple(self._records)
        diagnostics = tuple(self._diagnostics)
        if self._media == "image":
            return ImageSegmentationSnapshot(
                revision=self._revision,
                records=records,
                diagnostics=diagnostics,
                raw_output="".join(self._raw_output_parts),
            )
        return VideoSegmentationSnapshot(
            revision=self._revision,
            records=records,
            diagnostics=diagnostics,
            raw_output="".join(self._raw_output_parts),
        )

    def _result(self, outcome: ResponseStreamOutcome) -> SegmentationResult:
        records = tuple(self._records)
        diagnostics = tuple(self._diagnostics)
        if self._media == "image":
            return ImageSegmentationResult(
                revision=self._revision,
                records=records,
                diagnostics=diagnostics,
                raw_output="".join(self._raw_output_parts),
                outcome=outcome,
            )
        return VideoSegmentationResult(
            revision=self._revision,
            records=records,
            diagnostics=diagnostics,
            raw_output="".join(self._raw_output_parts),
            outcome=outcome,
        )


@dataclass(frozen=True, slots=True)
class _SegmentationFormat:
    _media: SegmentationMedia

    def create_parser(self) -> _SegmentationParser:
        return _SegmentationParser(self._media)


def image_segmentation_format() -> ResponseFormat[
    ImageSegmentationSnapshot, ImageSegmentationResult
]:
    """Create an image format whose parsers have isolated incremental state."""

    return cast(
        ResponseFormat[ImageSegmentationSnapshot, ImageSegmentationResult],
        _SegmentationFormat("image"),
    )


def video_segmentation_format() -> ResponseFormat[
    VideoSegmentationSnapshot, VideoSegmentationResult
]:
    """Create a video format whose parsers have isolated incremental state."""

    return cast(
        ResponseFormat[VideoSegmentationSnapshot, VideoSegmentationResult],
        _SegmentationFormat("video"),
    )
