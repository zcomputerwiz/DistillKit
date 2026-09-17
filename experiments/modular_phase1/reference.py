"""Independent recursive-descent interpreter for the scoped-bit language."""

from __future__ import annotations

from dataclasses import dataclass

from .language import Vocabulary, tokenize


@dataclass(frozen=True)
class Binding:
    use_position: int
    declaration_position: int
    declaration_value_position: int
    alternate_declaration_position: int
    name: str


@dataclass(frozen=True)
class Interpretation:
    answer: int
    answer_position: int
    bindings: tuple[Binding, ...]
    depths: tuple[int, ...]


class ReferenceInterpreter:
    """Parse a complete document and resolve names using lexical-scope dictionaries.

    This code does not call the specialist's streaming state machine. Its job is to be
    an independent oracle for generator checks, pointer labels, and evaluation answers.
    """

    def interpret(self, text: str) -> Interpretation:
        ids = tokenize(text)
        tokens = [Vocabulary.TOKENS[index] for index in ids]
        cursor = 1  # BOS
        scopes: list[dict[str, tuple[int, int, int]]] = []
        depths = [0] * len(tokens)
        bindings: list[Binding] = []
        query_value: int | None = None

        def skip_ws(position: int) -> int:
            while position < len(tokens) and tokens[position] in Vocabulary.WHITESPACE:
                depths[position] = len(scopes)
                position += 1
            return position

        def expect(position: int, wanted: str) -> int:
            position = skip_ws(position)
            if position >= len(tokens) or tokens[position] != wanted:
                got = tokens[position] if position < len(tokens) else "<end>"
                raise ValueError(f"expected {wanted!r} at token {position}, got {got!r}")
            depths[position] = len(scopes)
            return position + 1

        def resolve(name: str, use_position: int) -> int:
            candidates: list[tuple[int, int, int]] = []
            for scope in reversed(scopes):
                candidates.extend(scope.values())
                if name in scope:
                    value, declaration_position, value_position = scope[name]
                    alternates = [item[1] for item in candidates if item[1] != declaration_position]
                    alternate = alternates[0] if alternates else declaration_position
                    bindings.append(Binding(use_position, declaration_position, value_position,
                                            alternate, name))
                    return value
            raise ValueError(f"unbound variable {name!r} at token {use_position}")

        def operand(position: int) -> tuple[int, int]:
            position = skip_ws(position)
            token = tokens[position]
            depths[position] = len(scopes)
            if token in ("0", "1"):
                return int(token), position + 1
            if token in Vocabulary.VARIABLES:
                return resolve(token, position), position + 1
            raise ValueError(f"expected operand at token {position}, got {token!r}")

        while cursor < len(tokens):
            cursor = skip_ws(cursor)
            token = tokens[cursor]
            depths[cursor] = len(scopes)
            if token == "{":
                scopes.append({})
                depths[cursor] = len(scopes)
                cursor += 1
            elif token == "}":
                if not scopes:
                    raise ValueError(f"unmatched close at token {cursor}")
                scopes.pop()
                depths[cursor] = len(scopes)
                cursor += 1
            elif token == "let":
                if not scopes:
                    raise ValueError("declaration outside a scope")
                cursor += 1
                cursor = skip_ws(cursor)
                name_position = cursor
                name = tokens[cursor]
                if name not in Vocabulary.VARIABLES:
                    raise ValueError(f"invalid declaration name {name!r}")
                depths[cursor] = len(scopes)
                cursor += 1
                cursor = expect(cursor, "=")
                cursor = skip_ws(cursor)
                value_position = cursor
                if tokens[cursor] not in ("0", "1"):
                    raise ValueError("declaration value must be a bit")
                depths[cursor] = len(scopes)
                value = int(tokens[cursor])
                cursor += 1
                cursor = expect(cursor, ";")
                scopes[-1][name] = (value, name_position, value_position)
            elif token == "?":
                cursor += 1
                query_value, cursor = operand(cursor)
                while True:
                    probe = skip_ws(cursor)
                    if tokens[probe] != "^":
                        cursor = probe
                        break
                    depths[probe] = len(scopes)
                    right, cursor = operand(probe + 1)
                    query_value ^= right
                cursor = expect(cursor, ";")
            elif token == "=>":
                if query_value is None:
                    raise ValueError("answer before query")
                cursor += 1
                cursor = skip_ws(cursor)
                answer_position = cursor
                if tokens[cursor] not in ("0", "1"):
                    raise ValueError("answer must be one bit")
                if int(tokens[cursor]) != query_value:
                    raise ValueError("rendered answer disagrees with interpreted answer")
                depths[cursor] = len(scopes)
                cursor += 1
                cursor = skip_ws(cursor)
                if tokens[cursor] != "<eos>":
                    raise ValueError("trailing tokens after answer")
                depths[cursor] = len(scopes)
                return Interpretation(query_value, answer_position, tuple(bindings), tuple(depths))
            elif token == "<eos>":
                break
            else:
                raise ValueError(f"unexpected token {token!r} at {cursor}")
        raise ValueError("document has no answer")
