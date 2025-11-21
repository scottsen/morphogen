"""Tests for the parser."""

import pytest
from morphogen.parser.parser import parse
from morphogen.ast.nodes import *


def test_parse_assignment():
    """Test parsing simple assignment."""
    source = "x = 42"
    program = parse(source)

    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, Assignment)
    assert stmt.target == "x"
    assert isinstance(stmt.value, Literal)
    assert stmt.value.value == 42


def test_parse_typed_assignment():
    """Test parsing assignment with type annotation."""
    source = "x : f32 = 3.14"
    program = parse(source)

    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, Assignment)
    assert stmt.target == "x"
    assert stmt.type_annotation is not None
    assert stmt.type_annotation.base_type == "f32"


def test_parse_field_call():
    """Test parsing field method call."""
    source = "result = field.advect(x, v, dt)"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, Assignment)
    assert isinstance(stmt.value, Call)

    call = stmt.value
    assert isinstance(call.callee, FieldAccess)
    assert call.callee.field == "advect"
    assert len(call.args) == 3


def test_parse_decorator():
    """Test parsing decorator."""
    source = "@double_buffer\nx : Field2D<f32> = field.alloc(f32, [256, 256])"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, Assignment)
    assert len(stmt.decorators) == 1
    assert stmt.decorators[0].name == "double_buffer"


def test_parse_step():
    """Test parsing step block."""
    source = """
step {
  x = 42
  y = x + 1
}
"""
    program = parse(source)

    assert len(program.statements) == 1
    step = program.statements[0]
    assert isinstance(step, Step)
    assert len(step.body) == 2


def test_parse_binary_op():
    """Test parsing binary operations."""
    source = "result = a + b * c"
    program = parse(source)

    stmt = program.statements[0]
    expr = stmt.value

    # Should parse as: a + (b * c)
    assert isinstance(expr, BinaryOp)
    assert expr.operator == "+"
    assert isinstance(expr.left, Identifier)
    assert isinstance(expr.right, BinaryOp)
    assert expr.right.operator == "*"


def test_parse_function_call_kwargs():
    """Test parsing function call with keyword arguments."""
    source = "result = field.diffuse(x, rate=0.1, dt=0.01, method='cg')"
    program = parse(source)

    stmt = program.statements[0]
    call = stmt.value

    assert isinstance(call, Call)
    assert len(call.args) == 1
    assert len(call.kwargs) == 3
    assert "rate" in call.kwargs
    assert "dt" in call.kwargs
    assert "method" in call.kwargs
