from nemantix.core import Toolset

# import is needed because the toolset is not registered otherwise
# noinspection PyUnusedImports
from nemantix.stl.math_solver.base import MathSolverToolset


class TestSymbolicMath:
    # --- Simplification & Expansion ---

    def test_simplify_expression(self):
        """Test simplifying a complex algebraic expression."""
        ts = Toolset.get_tool(tool_name="MathSolverToolset.simplify_expression")
        # (x+1)^2 - (x^2 + 2x + 1) should be 0
        expr = "(x + 1)**2 - (x**2 + 2*x + 1)"
        result = ts(expr)
        assert result == "0"

    def test_expand_expression(self):
        """Test expanding a factored expression."""
        expr = "(x + 1) * (x - 1)"
        ts = Toolset.get_tool(tool_name="MathSolverToolset.expand_expression")
        result = ts(expr)
        # Expect x**2 - 1
        assert "x**2" in result
        assert "- 1" in result

    # --- Equation Solving ---

    def test_solve_equation_implicit_zero(self):
        """Test solving an equation assumed to equal zero (e.g., x^2 - 1)."""
        ts = Toolset.get_tool(tool_name="MathSolverToolset.solve_equation")
        # x^2 - 4 = 0 -> x = -2, 2
        result = ts("x**2 - 4", "x")
        assert "-2" in result
        assert "2" in result

    def test_solve_equation_explicit_equals(self):
        """Test solving an equation with '=' sign (e.g., x = 1)."""
        ts = Toolset.get_tool(tool_name="MathSolverToolset.solve_equation")
        # 2*x + 1 = 5 -> 2x = 4 -> x = 2
        result = ts("2*x + 1 = 5", "x")
        assert "[2]" in result or "2" in result

    def test_solve_no_solution_or_complex(self):
        """Test solving an equation with imaginary roots."""
        ts = Toolset.get_tool(tool_name="MathSolverToolset.solve_equation")
        # x^2 + 1 = 0 -> x = -i, i
        result = ts("x**2 + 1", "x")
        assert "I" in result  # SymPy represents imaginary unit as I

    # --- Calculus ---

    def test_derivative(self):
        """Test calculating derivatives."""
        # d/dx(x^3) = 3x^2
        ts = Toolset.get_tool(tool_name="MathSolverToolset.calculate_derivative")
        result = ts("x**3", "x")
        assert result == "3*x**2"

    def test_derivative_trig(self):
        """Test derivative of trigonometric functions."""
        # d/dx(sin(x)) = cos(x)
        ts = Toolset.get_tool(tool_name="MathSolverToolset.calculate_derivative")
        result = ts("sin(x)", "x")
        assert result == "cos(x)"

    def test_integral_indefinite(self):
        """Test indefinite integration."""
        # integral(2x) dx = x^2
        ts = Toolset.get_tool(tool_name="MathSolverToolset.calculate_integral")
        result = ts("2*x", "x")
        assert result == "x**2"

    def test_integral_definite(self):
        """Test definite integration with limits."""
        # integral from 0 to 3 of x^2 dx = [x^3/3] -> 27/3 - 0 = 9
        ts = Toolset.get_tool(tool_name="MathSolverToolset.calculate_integral")
        result = ts(expression="x**2", variable="x", lower_limit="0", upper_limit="3")
        assert result == "9"

    # --- Error Handling ---

    def test_solve_error_missing_variable(self):
        """Test solving for a variable that doesn't exist appropriately handles errors."""
        ts = Toolset.get_tool(tool_name="MathSolverToolset.solve_equation")
        result = ts("INVALID EQUATION ???", "x")
        assert "Error solving equation" in result
