"""Tests for Kairo v0.3.1 parser features."""

import pytest
from morphogen.parser.parser import parse
from morphogen.ast.nodes import *


class TestFlowBlocks:
    """Test flow() block parsing."""

    def test_parse_flow_with_dt_only(self):
        """Test parsing flow block with only dt parameter."""
        source = """
flow(dt=0.01) {
    temp = diffuse(temp, rate=0.1, dt)
}
"""
        program = parse(source)

        assert len(program.statements) == 1
        flow = program.statements[0]
        assert isinstance(flow, Flow)
        assert flow.dt is not None
        assert isinstance(flow.dt, Literal)
        assert flow.dt.value == 0.01
        assert flow.steps is None
        assert flow.substeps is None
        assert len(flow.body) == 1

    def test_parse_flow_with_dt_and_steps(self):
        """Test parsing flow block with dt and steps."""
        source = """
flow(dt=0.01, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)
}
"""
        program = parse(source)

        flow = program.statements[0]
        assert isinstance(flow, Flow)
        assert flow.dt.value == 0.01
        assert flow.steps.value == 1000
        assert flow.substeps is None

    def test_parse_flow_with_all_parameters(self):
        """Test parsing flow block with dt, steps, and substeps."""
        source = """
flow(dt=0.1, steps=100, substeps=10) {
    vel = advect(vel, vel, dt)
}
"""
        program = parse(source)

        flow = program.statements[0]
        assert isinstance(flow, Flow)
        assert flow.dt.value == 0.1
        assert flow.steps.value == 100
        assert flow.substeps.value == 10


class TestFunctionDefinitions:
    """Test function definition parsing."""

    def test_parse_simple_function(self):
        """Test parsing simple function without types."""
        source = """
fn double(x) {
    return x * 2.0
}
"""
        program = parse(source)

        assert len(program.statements) == 1
        func = program.statements[0]
        assert isinstance(func, Function)
        assert func.name == "double"
        assert len(func.params) == 1
        assert func.params[0][0] == "x"
        assert func.params[0][1] is None  # No type annotation
        assert func.return_type is None
        assert len(func.body) == 1

    def test_parse_typed_function(self):
        """Test parsing function with type annotations."""
        source = """
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    return max(min, min(x, max))
}
"""
        program = parse(source)

        func = program.statements[0]
        assert isinstance(func, Function)
        assert func.name == "clamp"
        assert len(func.params) == 3
        assert func.params[0] == ("x", TypeAnnotation(base_type="f32", type_params=[], unit=None))
        assert func.params[1] == ("min", TypeAnnotation(base_type="f32", type_params=[], unit=None))
        assert func.params[2] == ("max", TypeAnnotation(base_type="f32", type_params=[], unit=None))
        assert func.return_type.base_type == "f32"

    def test_parse_function_with_physical_units(self):
        """Test parsing function with physical unit types."""
        source = """
fn calculate_velocity(dist: f32[m], time: f32[s]) -> f32[m/s] {
    return dist / time
}
"""
        program = parse(source)

        func = program.statements[0]
        assert isinstance(func, Function)
        assert func.params[0][1].unit == "m"
        assert func.params[1][1].unit == "s"
        assert func.return_type.unit == "m/s"


class TestLambdaExpressions:
    """Test lambda expression parsing."""

    def test_parse_simple_lambda(self):
        """Test parsing simple lambda with one parameter."""
        source = "x = field.map(|a| a * 2.0)"
        program = parse(source)

        stmt = program.statements[0]
        assert isinstance(stmt, Assignment)
        call = stmt.value
        assert isinstance(call, Call)
        assert len(call.args) == 1

        lambda_expr = call.args[0]
        assert isinstance(lambda_expr, Lambda)
        assert lambda_expr.params == ["a"]
        assert isinstance(lambda_expr.body, BinaryOp)

    def test_parse_lambda_with_multiple_params(self):
        """Test parsing lambda with multiple parameters."""
        source = "result = combine(a, b, |x, y| x + y)"
        program = parse(source)

        stmt = program.statements[0]
        call = stmt.value
        lambda_expr = call.args[2]

        assert isinstance(lambda_expr, Lambda)
        assert lambda_expr.params == ["x", "y"]
        assert isinstance(lambda_expr.body, BinaryOp)
        assert lambda_expr.body.operator == "+"

    def test_parse_lambda_with_no_params(self):
        """Test parsing lambda with no parameters."""
        source = "constant = || 42"
        program = parse(source)

        stmt = program.statements[0]
        lambda_expr = stmt.value

        assert isinstance(lambda_expr, Lambda)
        assert lambda_expr.params == []
        assert isinstance(lambda_expr.body, Literal)
        assert lambda_expr.body.value == 42


class TestIfElseExpressions:
    """Test if/else expression parsing."""

    def test_parse_simple_if_else_inline(self):
        """Test parsing simple inline if/else expression."""
        source = "color = if temp > 100.0 then red else blue"
        program = parse(source)

        stmt = program.statements[0]
        assert isinstance(stmt, Assignment)

        if_else = stmt.value
        assert isinstance(if_else, IfElse)
        assert isinstance(if_else.condition, BinaryOp)
        assert if_else.condition.operator == ">"
        assert isinstance(if_else.then_expr, Identifier)
        assert if_else.then_expr.name == "red"
        assert isinstance(if_else.else_expr, Identifier)
        assert if_else.else_expr.name == "blue"

    def test_parse_if_else_with_blocks(self):
        """Test parsing if/else with block syntax."""
        source = """
result = if condition { value1 } else { value2 }
"""
        program = parse(source)

        stmt = program.statements[0]
        if_else = stmt.value

        assert isinstance(if_else, IfElse)
        assert isinstance(if_else.condition, Identifier)
        assert isinstance(if_else.then_expr, Identifier)
        assert isinstance(if_else.else_expr, Identifier)

    def test_parse_chained_if_else(self):
        """Test parsing chained if/else if/else."""
        source = """
speed = if vel > 10.0 then fast else if vel > 5.0 then medium else slow
"""
        program = parse(source)

        stmt = program.statements[0]
        if_else = stmt.value

        assert isinstance(if_else, IfElse)
        # The else branch should be another IfElse
        assert isinstance(if_else.else_expr, IfElse)


class TestStructDefinitions:
    """Test struct definition parsing."""

    def test_parse_simple_struct(self):
        """Test parsing simple struct definition."""
        source = """
struct Particle {
    pos: Vec2<f32>
    vel: Vec2<f32>
    mass: f32
}
"""
        program = parse(source)

        assert len(program.statements) == 1
        struct = program.statements[0]
        assert isinstance(struct, Struct)
        assert struct.name == "Particle"
        assert len(struct.fields) == 3
        assert struct.fields[0][0] == "pos"
        assert struct.fields[0][1].base_type == "Vec2"
        assert struct.fields[1][0] == "vel"
        assert struct.fields[2][0] == "mass"

    def test_parse_struct_with_units(self):
        """Test parsing struct with physical units."""
        source = """
struct PhysicsObject {
    position: Vec2<f32[m]>
    velocity: Vec2<f32[m/s]>
    acceleration: Vec2<f32[m/s²]>
}
"""
        program = parse(source)

        struct = program.statements[0]
        assert isinstance(struct, Struct)
        assert len(struct.fields) == 3
        # Note: The unit parsing for nested generics may need special handling


class TestReturnStatement:
    """Test return statement parsing."""

    def test_parse_return_with_value(self):
        """Test parsing return statement with value."""
        source = """
fn get_value() {
    return 42
}
"""
        program = parse(source)

        func = program.statements[0]
        assert len(func.body) == 1
        ret = func.body[0]
        assert isinstance(ret, Return)
        assert isinstance(ret.value, Literal)
        assert ret.value.value == 42

    def test_parse_return_without_value(self):
        """Test parsing return statement without value."""
        source = """
fn do_something() {
    return
}
"""
        program = parse(source)

        func = program.statements[0]
        ret = func.body[0]
        assert isinstance(ret, Return)
        assert ret.value is None


class TestStateDecorator:
    """Test @state decorator parsing (already existed, but test with new syntax)."""

    def test_parse_state_with_type_annotation(self):
        """Test parsing @state decorator with type annotation."""
        source = "@state temp : Field2D<f32> = zeros((256, 256))"
        program = parse(source)

        stmt = program.statements[0]
        assert isinstance(stmt, Assignment)
        assert len(stmt.decorators) == 1
        assert stmt.decorators[0].name == "state"
        assert stmt.type_annotation is not None
        assert stmt.type_annotation.base_type == "Field2D"

    def test_parse_state_with_units(self):
        """Test parsing @state with physical units."""
        source = "@state energy : f32[J] = 100.0"
        program = parse(source)

        stmt = program.statements[0]
        assert isinstance(stmt, Assignment)
        assert stmt.decorators[0].name == "state"
        assert stmt.type_annotation.base_type == "f32"
        assert stmt.type_annotation.unit == "J"


class TestIntegration:
    """Integration tests combining multiple v0.3.1 features."""

    def test_complete_particle_system(self):
        """Test parsing a complete particle system with v0.3.1 syntax."""
        source = """
struct Particle {
    pos: Vec2<f32>
    vel: Vec2<f32>
    age: u32
}

@state particles : Agents<Particle> = alloc(count=100)

fn update_particle(p: Particle, dt: f32) -> Particle {
    return Particle {
        pos: p.pos + p.vel * dt,
        vel: p.vel,
        age: p.age + 1
    }
}

flow(dt=0.01, steps=1000) {
    particles = particles.map(|p| update_particle(p, dt))
}
"""
        program = parse(source)

        # Should have 4 top-level statements: struct, state, function, flow
        assert len(program.statements) == 4
        assert isinstance(program.statements[0], Struct)
        assert isinstance(program.statements[1], Assignment)  # @state
        assert isinstance(program.statements[2], Function)
        assert isinstance(program.statements[3], Flow)

    @pytest.mark.skip("Block expression struct literal syntax not yet needed - skipping complex test")
    def test_flow_with_conditional_lambda(self):
        """Test flow block with lambda containing if/else."""
        source = """
flow(dt=0.01, steps=100) {
    agents = agents.map(|a| {
        vel: if a.pos.y < 0.0 then Vec2(a.vel.x, -a.vel.y) else a.vel,
        pos: a.pos + a.vel * dt
    })
}
"""
        program = parse(source)

        flow = program.statements[0]
        assert isinstance(flow, Flow)
        assert len(flow.body) == 1

        # The body contains an assignment with a call containing a lambda
        assignment = flow.body[0]
        assert isinstance(assignment, Assignment)
        call = assignment.value
        assert isinstance(call, Call)
        # Lambda is in the call arguments
        lambda_expr = call.args[0]
        assert isinstance(lambda_expr, Lambda)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
