"""
Tests for cross-domain composition parser support.
"""

from morphogen.parser.parser import Parser
from morphogen.lexer.lexer import Lexer
from morphogen.ast.nodes import Compose, Link, Identifier


def test_parse_compose():
    """Test parsing compose() statement."""
    source = """
    compose(module1, module2, module3)
    """

    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()

    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, Compose)
    assert len(stmt.modules) == 3
    assert all(isinstance(m, Identifier) for m in stmt.modules)


def test_parse_link_simple():
    """Test parsing simple link statement."""
    source = """
    link audio_module
    """

    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()

    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, Link)
    assert isinstance(stmt.target, Identifier)
    assert stmt.target.name == "audio_module"
    assert stmt.metadata is None


def test_parse_link_with_metadata():
    """Test parsing link statement with metadata."""
    source = """
    link physics_module {
        version: 1.0,
        required: true
    }
    """

    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()

    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, Link)
    assert isinstance(stmt.target, Identifier)
    assert stmt.target.name == "physics_module"
    assert stmt.metadata is not None
    assert "version" in stmt.metadata
    assert "required" in stmt.metadata


if __name__ == "__main__":
    test_parse_compose()
    print("✓ test_parse_compose passed")

    test_parse_link_simple()
    print("✓ test_parse_link_simple passed")

    test_parse_link_with_metadata()
    print("✓ test_parse_link_with_metadata passed")

    print("\nAll cross-domain parser tests passed!")
