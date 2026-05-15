import sympy
from typing import Optional
from nemantix.core import tool, Toolset


class MathSolverToolset(Toolset):
    """
    A toolset for performing advanced symbolic mathematical calculations
    using the SymPy library.
    """

    @tool
    def simplify_expression(self, expression: str) -> str:
        """
        Simplifies a mathematical expression algebraically.

        Args:
            expression (str): The mathematical expression to simplify.

        Returns:
            str: The simplified mathematical expression.

        Example call:
            simplify_expression(
                expression="(x + 1)**2 - (x**2 + 2*x + 1)"
            )
        """
        try:
            expr = sympy.sympify(expression)
            result = sympy.simplify(expr)
            return str(result)
        except Exception as e:
            return f"Error simplifying expression: {str(e)}"

    @tool
    def expand_expression(self, expression: str) -> str:
        """
        Expands a factored mathematical expression into a polynomial.

        Args:
            expression (str): The expression to expand.

        Returns:
            str: The expanded form of the expression.

        Example call:
            expand_expression(
                expression="(x + 3) * (x - 2)"
            )
        """
        try:
            expr = sympy.sympify(expression)
            result = sympy.expand(expr)
            return str(result)
        except Exception as e:
            return f"Error expanding expression: {str(e)}"

    @tool
    def solve_equation(self, equation: str, variable: str) -> str:
        """
        Solves an algebraic equation for a specific variable.

        Args:
            equation (str): The equation to solve. If no '=' is present, it assumes the expression equals zero.
            variable (str): The symbol to solve for.

        Returns:
            str: The list of solutions found for the variable.

        Example call:
            solve_equation(
                equation="x**2 - 5*x + 6",
                variable="x"
            )
        """
        try:
            # Handle standard "lhs = rhs" format by moving everything to lhs
            if "=" in equation:
                lhs, rhs = equation.split("=")
                eq_expr = sympy.sympify(lhs) - sympy.sympify(rhs)
            else:
                eq_expr = sympy.sympify(equation)

            target_var = sympy.Symbol(variable)
            solution = sympy.solve(eq_expr, target_var)
            return str(solution)
        except Exception as e:
            return f"Error solving equation: {str(e)}"

    @tool
    def calculate_derivative(self, expression: str, variable: str) -> str:
        """
        Calculates the symbolic derivative of an expression.

        Args:
            expression (str): The function to differentiate.
            variable (str): The variable with respect to which the derivative is taken.

        Returns:
            str: The derivative of the expression.

        Example call:
            calculate_derivative(
                expression="sin(x) * x**2",
                variable="x"
            )
        """
        try:
            expr = sympy.sympify(expression)
            var = sympy.Symbol(variable)
            result = sympy.diff(expr, var)
            return str(result)
        except Exception as e:
            return f"Error calculating derivative: {str(e)}"

    @tool
    def calculate_integral(
            self,
            expression: str,
            variable: str,
            lower_limit: Optional[str] = None,
            upper_limit: Optional[str] = None,
    ) -> str:
        """
        Calculates the integral of an expression. Performs indefinite integration if limits are omitted.

        Args:
            expression (str): The function to integrate.
            variable (str): The variable of integration.
            lower_limit (str, optional): The lower bound for definite integration. Defaults to None.
            upper_limit (str, optional): The upper bound for definite integration. Defaults to None.

        Returns:
            str: The result of the integration (symbolic or numeric).

        Example call:
            calculate_integral(
                expression="x**2",
                variable="x",
                lower_limit="0",
                upper_limit="3"
            )
        """
        try:
            expr = sympy.sympify(expression)
            var = sympy.Symbol(variable)

            if lower_limit is not None and upper_limit is not None:
                # Definite integral
                low = sympy.sympify(lower_limit)
                high = sympy.sympify(upper_limit)
                result = sympy.integrate(expr, (var, low, high))
            else:
                # Indefinite integral
                result = sympy.integrate(expr, var)

            return str(result)
        except Exception as e:
            return f"Error calculating integral: {str(e)}"
