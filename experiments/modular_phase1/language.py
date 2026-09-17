"""Synthetic scoped-bit language and deterministic corpus generation.

The generator and the interpreter intentionally live in separate modules.  Generation
stores the semantic program, while :mod:`reference` reparses rendered text and computes
answers without consulting generator state.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Iterable


class Vocabulary:
    """Frozen lexical vocabulary. Whitespace remains visible to the backbone."""

    TOKENS = (
        "<pad>", "<bos>", "<eos>", "{", "}", "let", "=", ";", "?", "^", "=>",
        "0", "1", "a", "b", "c", "d", "e", "f", "g", "h",
        "<sp1>", "<sp2>", "<tab>", "<nl>",
    )
    TO_ID = {token: index for index, token in enumerate(TOKENS)}
    PAD = TO_ID["<pad>"]
    BOS = TO_ID["<bos>"]
    EOS = TO_ID["<eos>"]
    VARIABLES = tuple("abcdefgh")
    WHITESPACE = {"<sp1>", "<sp2>", "<tab>", "<nl>"}

    @classmethod
    def hash(cls) -> str:
        payload = json.dumps(cls.TOKENS, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_LEXER = re.compile(r"=>|[{}=;?^]|[01]|let|[a-h]|[ \t\r\n]+")


def tokenize(text: str, *, add_special: bool = True) -> list[int]:
    """Tokenize exactly, rejecting every unrecognized byte instead of skipping it."""
    ids = [Vocabulary.BOS] if add_special else []
    offset = 0
    for match in _LEXER.finditer(text):
        if match.start() != offset:
            raise ValueError(f"unrecognized text at byte {offset}: {text[offset:match.start()]!r}")
        token = match.group(0)
        if token.isspace():
            if "\n" in token or "\r" in token:
                lexical = "<nl>"
            elif "\t" in token:
                lexical = "<tab>"
            elif len(token) == 1:
                lexical = "<sp1>"
            else:
                lexical = "<sp2>"
        else:
            lexical = token
        ids.append(Vocabulary.TO_ID[lexical])
        offset = match.end()
    if offset != len(text):
        raise ValueError(f"unrecognized text at byte {offset}: {text[offset:]!r}")
    if add_special:
        ids.append(Vocabulary.EOS)
    return ids


def decode(ids: Iterable[int]) -> list[str]:
    return [Vocabulary.TOKENS[int(index)] for index in ids]


@dataclass(frozen=True)
class Declaration:
    depth: int
    name: str
    value: int


@dataclass(frozen=True)
class Program:
    depth: int
    declarations: tuple[Declaration, ...]
    query: tuple[str | int, ...]
    literal: bool

    def canonical(self, *, normalize_names: bool = True) -> str:
        if normalize_names:
            mapping: dict[str, str] = {}

            def name(value: str) -> str:
                if value not in mapping:
                    mapping[value] = f"v{len(mapping)}"
                return mapping[value]
        else:
            name = lambda value: value  # noqa: E731
        declarations = [
            [item.depth, name(item.name), item.value] for item in self.declarations
        ]
        query = [name(item) if isinstance(item, str) else item for item in self.query]
        return json.dumps(
            {"depth": self.depth, "declarations": declarations, "query": query,
             "literal": self.literal},
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass
class Document:
    document_id: str
    text: str
    token_ids: list[int]
    answer: int
    answer_position: int
    program: Program
    metadata: dict = field(default_factory=dict)

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(self.program.canonical().encode("utf-8")).hexdigest()

    def to_json(self) -> dict:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "token_ids": self.token_ids,
            "answer": self.answer,
            "answer_position": self.answer_position,
            "program": {
                "depth": self.program.depth,
                "declarations": [item.__dict__ for item in self.program.declarations],
                "query": list(self.program.query),
                "literal": self.program.literal,
            },
            "metadata": self.metadata,
            "canonical_hash": self.canonical_hash,
        }

    @classmethod
    def from_json(cls, value: dict) -> "Document":
        program = value["program"]
        return cls(
            document_id=value["document_id"],
            text=value["text"],
            token_ids=[int(item) for item in value["token_ids"]],
            answer=int(value["answer"]),
            answer_position=int(value["answer_position"]),
            program=Program(
                depth=int(program["depth"]),
                declarations=tuple(Declaration(**item) for item in program["declarations"]),
                query=tuple(program["query"]),
                literal=bool(program["literal"]),
            ),
            metadata=dict(value["metadata"]),
        )


def _space(rng: random.Random, style: str, required: bool = False) -> str:
    if style == "spaces":
        return " " * rng.randint(1, 3) if required or rng.random() < 0.8 else ""
    if style == "tabs":
        return "\t" if required or rng.random() < 0.75 else ""
    if style == "newlines":
        return rng.choice(["\n", "\n  ", "\r\n\t"]) if required or rng.random() < 0.7 else ""
    if style == "mixed":
        return rng.choice([" ", "  ", "\t", "\n", "\n  "]) if required or rng.random() < 0.8 else ""
    raise ValueError(f"unknown whitespace style {style!r}")


def render_program(program: Program, answer: int, style: str, seed: int) -> str:
    rng = random.Random(seed)
    semantic: list[str] = []
    if not program.literal:
        declarations = list(program.declarations)
        by_depth = {depth: [] for depth in range(1, program.depth + 1)}
        for declaration in declarations:
            by_depth[declaration.depth].append(declaration)
        for depth in range(1, program.depth + 1):
            semantic.append("{")
            for declaration in by_depth[depth]:
                semantic.extend(["let", declaration.name, "=", str(declaration.value), ";"])
    semantic.append("?")
    semantic.append(str(program.query[0]))
    for operand in program.query[1:]:
        semantic.extend(["^", str(operand)])
    semantic.append(";")
    if not program.literal:
        semantic.extend("}" for _ in range(program.depth))
    semantic.extend(["=>", str(answer)])

    chunks: list[str] = []
    for index, token in enumerate(semantic):
        if index:
            previous = semantic[index - 1]
            required = previous in ("let",) or (
                (previous in Vocabulary.VARIABLES or previous in ("0", "1"))
                and (token in Vocabulary.VARIABLES or token in ("0", "1", "let"))
            )
            chunks.append(_space(rng, style, required=required))
        chunks.append(token)
    return "".join(chunks)


def _resolve(program: Program, operand: str | int) -> int:
    if isinstance(operand, int):
        return operand
    active = [item for item in program.declarations if item.name == operand]
    if not active:
        raise ValueError(f"unbound generated variable {operand}")
    deepest = max(item.depth for item in active)
    # Within one scope, the most recent declaration shadows earlier declarations too.
    return next(item.value for item in reversed(active) if item.depth == deepest)


def _sample_program(
    rng: random.Random,
    *,
    literal_fraction: float,
    depth_range: tuple[int, int],
    history_range: tuple[int, int],
    force_heldout_combo: bool,
    force_literal: bool | None,
) -> Program:
    literal = rng.random() < literal_fraction if force_literal is None else force_literal
    if literal:
        length = 1 if rng.random() < 0.5 else rng.randint(2, 8)
        query = tuple(rng.randint(0, 1) for _ in range(length))
        return Program(depth=0, declarations=(), query=query, literal=True)

    depth = 4 if force_heldout_combo else rng.randint(*depth_range)
    target_count = max(depth, rng.randint(*history_range))
    declarations: list[Declaration] = []
    active_names: list[str] = []
    for level in range(1, depth + 1):
        if level == 1 or rng.random() < 0.55:
            fresh = [name for name in Vocabulary.VARIABLES if name not in active_names]
            chosen = rng.choice(fresh or list(Vocabulary.VARIABLES))
        else:
            chosen = rng.choice(active_names)
        declarations.append(Declaration(level, chosen, rng.randint(0, 1)))
        if chosen not in active_names:
            active_names.append(chosen)
    while len(declarations) < target_count:
        level = rng.randint(1, depth)
        if rng.random() < 0.45 and active_names:
            chosen = rng.choice(active_names)
        else:
            chosen = rng.choice(Vocabulary.VARIABLES)
            if chosen not in active_names:
                active_names.append(chosen)
        declarations.append(Declaration(level, chosen, rng.randint(0, 1)))
    declarations.sort(key=lambda item: item.depth)
    visible = sorted({item.name for item in declarations})
    if force_heldout_combo:
        # Make the held-out conjunction explicit: depth 4, XOR, at least two shadows.
        base = visible[0]
        declarations.extend([
            Declaration(3, base, rng.randint(0, 1)),
            Declaration(4, base, rng.randint(0, 1)),
        ])
        declarations.sort(key=lambda item: item.depth)
        visible = sorted({item.name for item in declarations})
        query = (base, rng.choice(visible))
    else:
        query = (rng.choice(visible),)
        if rng.random() < 0.55:
            query = (query[0], rng.choice(visible))
    return Program(depth=depth, declarations=tuple(declarations), query=query, literal=False)


def generate_document(
    seed: int,
    document_index: int,
    *,
    split: str,
    literal_fraction: float = 0.20,
    depth_range: tuple[int, int] = (1, 4),
    history_range: tuple[int, int] = (4, 12),
    whitespace_styles: tuple[str, ...] = ("spaces", "tabs", "newlines", "mixed"),
    force_heldout_combo: bool = False,
    force_literal: bool | None = None,
) -> Document:
    mixed_seed = (seed * 1_000_003 + document_index * 97_409 + 17) & 0xFFFFFFFF
    rng = random.Random(mixed_seed)
    program = _sample_program(
        rng,
        literal_fraction=literal_fraction,
        depth_range=depth_range,
        history_range=history_range,
        force_heldout_combo=force_heldout_combo,
        force_literal=force_literal,
    )
    values = [_resolve(program, item) for item in program.query]
    answer = values[0]
    for value in values[1:]:
        answer ^= value
    style = "newlines" if force_heldout_combo else rng.choice(whitespace_styles)
    text = render_program(program, answer, style, mixed_seed ^ 0xBAD5EED)
    ids = tokenize(text)
    # The answer is the only lexical bit between the arrow and EOS.
    arrow = max(index for index, token in enumerate(ids) if token == Vocabulary.TO_ID["=>"])
    answer_position = arrow + 1
    if ids[answer_position] not in (Vocabulary.TO_ID["0"], Vocabulary.TO_ID["1"]):
        # Whitespace can intervene, so locate the next non-whitespace token.
        answer_position = next(
            index for index in range(arrow + 1, len(ids))
            if Vocabulary.TOKENS[ids[index]] not in Vocabulary.WHITESPACE
        )
    shadow_count = len(program.declarations) - len({item.name for item in program.declarations})
    kind = ("literal_xor" if len(program.query) > 1 else "literal") if program.literal else (
        "xor" if len(program.query) > 1 else "lookup"
    )
    identifier = hashlib.sha256(
        f"{split}:{seed}:{document_index}:{program.canonical(normalize_names=False)}".encode()
    ).hexdigest()[:20]
    return Document(
        document_id=f"{split}-{identifier}",
        text=text,
        token_ids=ids,
        answer=answer,
        answer_position=answer_position,
        program=program,
        metadata={
            "split": split,
            "depth": program.depth,
            "history": len(program.declarations),
            "query_kind": kind,
            "literal": program.literal,
            "shadow_count": shadow_count,
            "whitespace": style,
            "heldout_combo": force_heldout_combo,
            "variant": "base",
        },
    )


def rename_document(document: Document, shift: int = 3) -> Document:
    mapping = {
        name: Vocabulary.VARIABLES[(index + shift) % len(Vocabulary.VARIABLES)]
        for index, name in enumerate(Vocabulary.VARIABLES)
    }
    program = Program(
        depth=document.program.depth,
        declarations=tuple(
            Declaration(item.depth, mapping[item.name], item.value)
            for item in document.program.declarations
        ),
        query=tuple(mapping[item] if isinstance(item, str) else item
                    for item in document.program.query),
        literal=document.program.literal,
    )
    style = document.metadata["whitespace"]
    text = render_program(program, document.answer, style, seed=991 + shift)
    ids = tokenize(text)
    arrow = max(index for index, token in enumerate(ids) if token == Vocabulary.TO_ID["=>"])
    answer_position = next(
        index for index in range(arrow + 1, len(ids))
        if Vocabulary.TOKENS[ids[index]] not in Vocabulary.WHITESPACE
    )
    metadata = dict(document.metadata, variant="renamed", base_id=document.document_id)
    return Document(document.document_id + "-renamed", text, ids, document.answer,
                    answer_position, program, metadata)


def whitespace_document(document: Document, style: str) -> Document:
    text = render_program(document.program, document.answer, style, seed=1776)
    ids = tokenize(text)
    arrow = max(index for index, token in enumerate(ids) if token == Vocabulary.TO_ID["=>"])
    answer_position = next(
        index for index in range(arrow + 1, len(ids))
        if Vocabulary.TOKENS[ids[index]] not in Vocabulary.WHITESPACE
    )
    metadata = dict(document.metadata, whitespace=style, variant="whitespace",
                    base_id=document.document_id)
    return Document(document.document_id + f"-ws-{style}", text, ids, document.answer,
                    answer_position, document.program, metadata)
