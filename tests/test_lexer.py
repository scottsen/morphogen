"""Tests for the lexer."""

import pytest
from morphogen.lexer.lexer import Lexer, TokenType


def test_tokenize_numbers():
    """Test number tokenization."""
    lexer = Lexer("42 3.14 1.5e-10")
    tokens = lexer.tokenize()

    assert len(tokens) == 4  # 3 numbers + EOF
    assert tokens[0].type == TokenType.NUMBER
    assert tokens[0].value == 42
    assert tokens[1].type == TokenType.NUMBER
    assert tokens[1].value == 3.14
    assert tokens[2].type == TokenType.NUMBER
    assert tokens[2].value == 1.5e-10


def test_tokenize_identifiers():
    """Test identifier tokenization."""
    lexer = Lexer("foo bar_baz _private")
    tokens = lexer.tokenize()

    assert len(tokens) == 4  # 3 identifiers + EOF
    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "foo"
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == "bar_baz"
    assert tokens[2].type == TokenType.IDENTIFIER
    assert tokens[2].value == "_private"


def test_tokenize_keywords():
    """Test keyword tokenization."""
    lexer = Lexer("step substep module compose")
    tokens = lexer.tokenize()

    assert len(tokens) == 5  # 4 keywords + EOF
    assert tokens[0].type == TokenType.STEP
    assert tokens[1].type == TokenType.SUBSTEP
    assert tokens[2].type == TokenType.MODULE
    assert tokens[3].type == TokenType.COMPOSE


def test_tokenize_operators():
    """Test operator tokenization."""
    lexer = Lexer("+ - * / = == != < <= > >=")
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.PLUS
    assert tokens[1].type == TokenType.MINUS
    assert tokens[2].type == TokenType.STAR
    assert tokens[3].type == TokenType.SLASH
    assert tokens[4].type == TokenType.ASSIGN
    assert tokens[5].type == TokenType.EQ
    assert tokens[6].type == TokenType.NE
    assert tokens[7].type == TokenType.LT
    assert tokens[8].type == TokenType.LE
    assert tokens[9].type == TokenType.GT
    assert tokens[10].type == TokenType.GE


def test_tokenize_strings():
    """Test string tokenization."""
    lexer = Lexer('"hello" "world with spaces" "escape\\ntest"')
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "hello"
    assert tokens[1].type == TokenType.STRING
    assert tokens[1].value == "world with spaces"
    assert tokens[2].type == TokenType.STRING
    assert tokens[2].value == "escape\ntest"


def test_tokenize_field_access():
    """Test field access tokenization."""
    lexer = Lexer("field.advect(x, v, dt)")
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "field"
    assert tokens[1].type == TokenType.DOT
    assert tokens[2].type == TokenType.IDENTIFIER
    assert tokens[2].value == "advect"
    assert tokens[3].type == TokenType.LPAREN


def test_skip_comments():
    """Test comment skipping."""
    lexer = Lexer("foo # this is a comment\nbar")
    tokens = lexer.tokenize()

    # Should have: foo, newline, bar, EOF
    assert len(tokens) == 4
    assert tokens[0].value == "foo"
    assert tokens[1].type == TokenType.NEWLINE
    assert tokens[2].value == "bar"


def test_decorator():
    """Test decorator tokenization."""
    lexer = Lexer("@double_buffer")
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.AT
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == "double_buffer"


def test_type_annotation():
    """Test type annotation tokenization."""
    lexer = Lexer("x : Field2D<f32[m/s]>")
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "x"
    assert tokens[1].type == TokenType.COLON
    assert tokens[2].type == TokenType.IDENTIFIER
    assert tokens[2].value == "Field2D"
    assert tokens[3].type == TokenType.LT
    assert tokens[4].type == TokenType.IDENTIFIER
    assert tokens[4].value == "f32"
