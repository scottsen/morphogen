"""Tests for the use statement functionality."""

import pytest
from morphogen.lexer.lexer import Lexer, TokenType
from morphogen.parser.parser import Parser
from morphogen.ast.nodes import Use, Program
from morphogen.runtime.runtime import Runtime, ExecutionContext
from morphogen.core.domain_registry import DomainRegistry


class TestUseLexer:
    """Test lexer tokenization of use statements."""

    def test_tokenize_use_keyword(self):
        """Test that 'use' is recognized as a keyword."""
        lexer = Lexer("use field")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.USE
        assert tokens[0].value == "use"
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "field"

    def test_tokenize_use_multiple_domains(self):
        """Test tokenizing use with multiple domains."""
        lexer = Lexer("use field, visual, agent")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.USE
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "field"
        assert tokens[2].type == TokenType.COMMA
        assert tokens[3].type == TokenType.IDENTIFIER
        assert tokens[3].value == "visual"
        assert tokens[4].type == TokenType.COMMA
        assert tokens[5].type == TokenType.IDENTIFIER
        assert tokens[5].value == "agent"


class TestUseParser:
    """Test parser parsing of use statements."""

    def test_parse_single_domain(self):
        """Test parsing use statement with single domain."""
        source = "use field"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Use)
        assert stmt.domains == ["field"]
        assert stmt.aliases == {}

    def test_parse_multiple_domains(self):
        """Test parsing use statement with multiple domains."""
        source = "use field, visual, agent"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Use)
        assert stmt.domains == ["field", "visual", "agent"]
        assert stmt.aliases == {}

    def test_parse_use_with_newlines(self):
        """Test parsing use statement with newlines after commas."""
        source = """use field,
            visual,
            agent"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 1
        stmt = program.statements[0]
        assert isinstance(stmt, Use)
        assert stmt.domains == ["field", "visual", "agent"]

    def test_parse_use_at_start_of_program(self):
        """Test use statement at the beginning of a program."""
        source = """use field, visual

temp = field.alloc((128, 128), fill_value=0.0)
vis = visual.colorize(temp, palette="fire")
"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 3
        assert isinstance(program.statements[0], Use)
        assert program.statements[0].domains == ["field", "visual"]


class TestUseRuntime:
    """Test runtime execution of use statements."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear and reinitialize domain registry for each test
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_execute_use_valid_domain(self):
        """Test executing use statement with valid domain."""
        source = "use field"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Should not raise any exceptions
        for stmt in program.statements:
            runtime.execute_statement(stmt)

    def test_execute_use_multiple_valid_domains(self):
        """Test executing use statement with multiple valid domains."""
        source = "use field, visual, agent"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Should not raise any exceptions
        for stmt in program.statements:
            runtime.execute_statement(stmt)

    def test_execute_use_invalid_domain(self):
        """Test executing use statement with invalid domain."""
        source = "use nonexistent_domain"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Should raise ValueError for invalid domain
        with pytest.raises(ValueError) as exc_info:
            for stmt in program.statements:
                runtime.execute_statement(stmt)

        assert "nonexistent_domain" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_execute_use_one_invalid_among_valid(self):
        """Test executing use statement with one invalid domain among valid ones."""
        source = "use field, invalid_domain, visual"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Should raise ValueError for invalid domain
        with pytest.raises(ValueError) as exc_info:
            for stmt in program.statements:
                runtime.execute_statement(stmt)

        assert "invalid_domain" in str(exc_info.value)


class TestUseIntegration:
    """Integration tests for use statement with actual domain operations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear and reinitialize domain registry for each test
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_use_with_field_operations(self):
        """Test use statement followed by field operations."""
        source = """use field

temp = field.alloc((64, 64), fill_value=0.5)
temp = field.diffuse(temp, rate=0.1, dt=0.01, iterations=5)
"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Execute all statements
        for stmt in program.statements:
            runtime.execute_statement(stmt)

        # Verify temp was created and has correct shape
        temp = runtime.context.get_variable('temp')
        assert temp.shape == (64, 64)

    def test_use_with_visual_operations(self):
        """Test use statement followed by visual operations."""
        source = """use field, visual

temp = field.alloc((32, 32), fill_value=0.7)
vis = visual.colorize(temp, palette="viridis")
"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Execute all statements
        for stmt in program.statements:
            runtime.execute_statement(stmt)

        # Verify variables were created
        temp = runtime.context.get_variable('temp')
        vis = runtime.context.get_variable('vis')
        assert temp.shape == (32, 32)
        assert vis.shape == (32, 32)

    def test_use_with_agents_operations(self):
        """Test use statement with agent domain operations."""
        source = """use field, agent

positions = field.alloc((10, 2), fill_value=0.0)
velocities = field.alloc((10, 2), fill_value=0.0)
"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Execute all statements
        for stmt in program.statements:
            runtime.execute_statement(stmt)

        # Verify variables were created
        positions = runtime.context.get_variable('positions')
        velocities = runtime.context.get_variable('velocities')
        assert positions.shape == (10, 2)
        assert velocities.shape == (10, 2)

    def test_multiple_use_statements(self):
        """Test multiple use statements in a program."""
        source = """use field
use visual
use agent

temp = field.alloc((16, 16), fill_value=1.0)
vis = visual.colorize(temp, palette="fire")
"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 5
        assert isinstance(program.statements[0], Use)
        assert isinstance(program.statements[1], Use)
        assert isinstance(program.statements[2], Use)

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Execute all statements
        for stmt in program.statements:
            runtime.execute_statement(stmt)

    def test_use_without_domain_still_works(self):
        """Test that programs work without use statement (backwards compatibility)."""
        source = """temp = field.alloc((16, 16), fill_value=1.0)
vis = visual.colorize(temp, palette="fire")
"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        ctx = ExecutionContext()
        runtime = Runtime(ctx)

        # Should still work without use statement (domains are pre-registered)
        for stmt in program.statements:
            runtime.execute_statement(stmt)

        temp = runtime.context.get_variable('temp')
        vis = runtime.context.get_variable('vis')
        assert temp.shape == (16, 16)
        assert vis.shape == (16, 16)


class TestUseDocumentation:
    """Test that use statement serves as documentation."""

    def test_use_statement_documents_dependencies(self):
        """Test that use statements clearly document domain dependencies."""
        # Example program with clear dependency documentation
        source = """use field, visual, agent

# Now it's clear this program uses field, visual, and agent domains
temp = field.alloc((128, 128), fill_value=0.0)
vis = visual.colorize(temp, palette="fire")
"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        # Extract use statements to see dependencies
        use_stmts = [s for s in program.statements if isinstance(s, Use)]
        all_domains = []
        for use_stmt in use_stmts:
            all_domains.extend(use_stmt.domains)

        assert "field" in all_domains
        assert "visual" in all_domains
        assert "agent" in all_domains
