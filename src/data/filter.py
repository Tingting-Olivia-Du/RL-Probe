"""
Problem Filtering Utilities

Filter MATH problems based on model performance and other criteria.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import sympy
    from sympy import simplify, sympify, N, Symbol, symbols
    # Try to import LaTeX parser (requires antlr4-python3-runtime)
    try:
        from sympy.parsing.latex import parse_latex
        LATEX_PARSER_AVAILABLE = True
    except ImportError:
        LATEX_PARSER_AVAILABLE = False
        logger.warning("SymPy LaTeX parser not available. Install antlr4-python3-runtime for LaTeX parsing.")
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    LATEX_PARSER_AVAILABLE = False
    logger.warning("SymPy not available. Mathematical equivalence checking will be disabled.")


class ProblemFilter:
    """
    Filter problems based on model evaluation results.

    Identifies problems where:
    - DPO model produces incorrect answers
    - RLVR model produces correct answers
    """

    def __init__(
        self, 
        answer_extraction_pattern: Optional[str] = None,
        use_sympy: bool = True,
    ):
        """
        Initialize the problem filter.

        Args:
            answer_extraction_pattern: Regex pattern for extracting answers
            use_sympy: Whether to use SymPy for mathematical equivalence checking
        """
        # Default pattern matches \\boxed{...} format used in MATH
        self.answer_pattern = answer_extraction_pattern or r"\\boxed\{([^}]+)\}"
        self.use_sympy = use_sympy and SYMPY_AVAILABLE
        if self.use_sympy:
            logger.info("SymPy-based equivalence checking enabled")
        else:
            logger.info("Using string-based comparison (SymPy disabled or unavailable)")

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the final answer from a model response.

        Args:
            response: Model's generated response

        Returns:
            Extracted answer string or None
        """
        matches = re.findall(self.answer_pattern, response)
        if matches:
            return matches[-1].strip()  # Return last match (final answer)
        return None

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.

        Handles common variations in mathematical notation.
        """
        if answer is None:
            return ""

        # Remove whitespace
        answer = answer.strip()

        # Normalize common mathematical expressions
        answer = answer.replace(" ", "")
        answer = answer.replace("\\frac", "frac")
        answer = answer.replace("\\sqrt", "sqrt")
        answer = answer.replace("\\pi", "pi")
        answer = answer.replace("\\cdot", "*")

        return answer.lower()

    def _parse_math_expression(self, expr: str) -> Optional[sympy.Expr]:
        """
        Parse a mathematical expression from LaTeX or text format.

        Args:
            expr: Mathematical expression string

        Returns:
            SymPy expression or None if parsing fails
        """
        if not SYMPY_AVAILABLE:
            return None

        try:
            # Try parsing as LaTeX first (if parser available)
            if LATEX_PARSER_AVAILABLE:
                try:
                    return parse_latex(expr)
                except Exception:
                    pass

            # Try parsing as standard SymPy expression
            # Replace common LaTeX commands with SymPy equivalents
            cleaned = expr.strip()
            
            # Handle LaTeX fractions: \frac{a}{b} -> (a)/(b)
            frac_pattern = r"\\frac\{([^}]+)\}\{([^}]+)\}"
            while re.search(frac_pattern, cleaned):
                cleaned = re.sub(frac_pattern, r"(\1)/(\2)", cleaned)
            
            # Handle LaTeX sqrt: \sqrt{x} -> sqrt(x)
            sqrt_pattern = r"\\sqrt\{([^}]+)\}"
            cleaned = re.sub(sqrt_pattern, r"sqrt(\1)", cleaned)
            
            # Handle LaTeX sqrt with index: \sqrt[n]{x} -> root(x, n)
            sqrtn_pattern = r"\\sqrt\[([^\]]+)\]\{([^}]+)\}"
            cleaned = re.sub(sqrtn_pattern, r"root(\2, \1)", cleaned)
            
            # Replace other LaTeX commands
            replacements = {
                "\\pi": "pi",
                "\\cdot": "*",
                "\\times": "*",
                "\\div": "/",
                "\\pm": "+-",
                "\\mp": "-+",
                "\\leq": "<=",
                "\\geq": ">=",
                "\\neq": "!=",
                "\\approx": "~",
                "\\infty": "oo",
                "\\sum": "Sum",
                "\\prod": "Product",
                "\\int": "Integral",
            }
            for latex_cmd, sympy_cmd in replacements.items():
                cleaned = cleaned.replace(latex_cmd, sympy_cmd)
            
            # Remove extra spaces
            cleaned = re.sub(r"\s+", "", cleaned)
            
            try:
                return sympify(cleaned, evaluate=False)
            except Exception:
                pass

            # Try direct sympify as last resort
            return sympify(expr, evaluate=False)
        except Exception as e:
            logger.debug(f"Failed to parse expression '{expr}': {e}")
            return None

    def _check_sympy_equivalence(
        self,
        expr1: str,
        expr2: str,
        tolerance: float = 1e-10,
    ) -> bool:
        """
        Check if two mathematical expressions are equivalent using SymPy.

        Args:
            expr1: First expression
            expr2: Second expression
            tolerance: Numerical tolerance for comparison

        Returns:
            True if expressions are equivalent, False otherwise
        """
        if not SYMPY_AVAILABLE:
            return False

        try:
            parsed1 = self._parse_math_expression(expr1)
            parsed2 = self._parse_math_expression(expr2)

            if parsed1 is None or parsed2 is None:
                return False

            # Simplify both expressions
            simplified1 = simplify(parsed1)
            simplified2 = simplify(parsed2)

            # Try symbolic equality first
            if simplified1.equals(simplified2):
                return True

            # Try numerical comparison if both are numeric
            try:
                num1 = float(N(simplified1))
                num2 = float(N(simplified2))
                return abs(num1 - num2) < tolerance
            except:
                pass

            # Try subtracting and checking if result simplifies to zero
            diff = simplify(simplified1 - simplified2)
            if diff.equals(0):
                return True

            # Try numerical evaluation of difference
            try:
                diff_val = float(N(diff))
                return abs(diff_val) < tolerance
            except:
                pass

            return False
        except Exception as e:
            logger.debug(f"SymPy equivalence check failed: {e}")
            return False

    def check_correctness(
        self,
        model_response: str,
        ground_truth: str,
    ) -> bool:
        """
        Check if model response matches ground truth.

        Uses SymPy for mathematical equivalence checking if available,
        otherwise falls back to string-based comparison.

        Args:
            model_response: Model's generated response
            ground_truth: Ground truth answer

        Returns:
            True if answers match, False otherwise
        """
        extracted = self.extract_answer(model_response)
        if extracted is None:
            return False

        # Try SymPy-based equivalence checking first if enabled
        if self.use_sympy:
            is_equivalent = self._check_sympy_equivalence(extracted, ground_truth)
            if is_equivalent:
                return True

        # Fall back to string-based comparison
        norm_extracted = self.normalize_answer(extracted)
        norm_truth = self.normalize_answer(ground_truth)

        return norm_extracted == norm_truth

    def evaluate_model_responses(
        self,
        problems: List[Dict[str, Any]],
        responses: List[str],
    ) -> Dict[str, bool]:
        """
        Evaluate a batch of model responses.

        Args:
            problems: List of problem dictionaries with ground truth
            responses: List of model responses

        Returns:
            Dict mapping problem identifiers to correctness
        """
        results = {}

        for problem, response in zip(problems, responses):
            problem_id = problem.get("unique_id", str(hash(problem["problem"])))
            ground_truth = problem.get("solution", problem.get("answer", ""))

            # Extract ground truth answer
            gt_answer = self.extract_answer(ground_truth)
            if gt_answer is None:
                gt_answer = ground_truth  # Use raw if no boxed format

            is_correct = self.check_correctness(response, gt_answer)
            results[problem_id] = is_correct

        return results


class DifficultyAnalyzer:
    """Analyze problem difficulty based on various metrics."""

    @staticmethod
    def estimate_steps(solution: str) -> int:
        """
        Estimate number of reasoning steps in a solution.

        Args:
            solution: Ground truth solution text

        Returns:
            Estimated number of steps
        """
        # Count step indicators
        step_patterns = [
            r"step \d+",
            r"first|second|third|fourth|fifth",
            r"then|next|finally",
            r"therefore|thus|hence",
            r"we have|we get|we find",
        ]

        step_count = 0
        solution_lower = solution.lower()

        for pattern in step_patterns:
            step_count += len(re.findall(pattern, solution_lower))

        # Also count equation lines
        equation_count = solution.count("=")

        return max(step_count, equation_count // 2, 1)

    @staticmethod
    def categorize_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize a problem by type and difficulty.

        Args:
            problem: Problem dictionary

        Returns:
            Dictionary with category information
        """
        problem_text = problem.get("problem", "")
        solution = problem.get("solution", "")

        categories = {
            "has_algebra": bool(re.search(r"solve|equation|variable", problem_text.lower())),
            "has_geometry": bool(re.search(r"triangle|circle|angle|area", problem_text.lower())),
            "has_probability": bool(re.search(r"probability|chance|random", problem_text.lower())),
            "has_combinatorics": bool(re.search(r"ways|arrange|choose|combination", problem_text.lower())),
            "has_number_theory": bool(re.search(r"divisible|prime|remainder|factor", problem_text.lower())),
            "estimated_steps": DifficultyAnalyzer.estimate_steps(solution),
            "solution_length": len(solution),
        }

        return categories
