"""This file add the console interface to the package."""
import argparse
from typing import Any, List, Optional

from better_experimentation.__init__ import BetterExperimentation


def parse_args(args: Optional[List[Any]] = None) -> argparse.Namespace:
    """Parse the command line arguments for the `better_experimentation` binary.

    Args:
      args: List of input arguments. (Default value=None).

    Returns:
      Namespace with parsed arguments.

    """
    parser = argparse.ArgumentParser(
        description="Apply continuous experimentation to compare models and generate a final report in HTML"
    )

    # Better Experimentation Variables
    parser.add_argument(
        "models_trained_path",
        type=str,
        help="Path with saved trained models to apply continuous experimentation",
    )

    parser.add_argument(
        "x_test_path",
        type=str,
        help="CSV file (or other file type supported by pandas) that represent X axis data (features) to generate scores to apply continuous experimentation",
    )

    parser.add_argument(
        "y_test_path",
        type=str,
        help="CSV file (or other file type supported by pandas) that represent Y axis data (target) to generate scores to apply continuous experimentation",
    )

    parser.add_argument(
        "scores_target",
        type=str,
        help="Score target to use like a reference to define best model and statistical details during continuous experimentation. Possible Values: ACCURACY, PRECISION, RECALL, MAE, MSE",
    )

    parser.add_argument(
        "--n_splits",
        type=str,
        help="Number of splits to generate cases of tests to apply continuous experimentation. Default value = 100" ,
        default=100,
    )

    parser.add_argument(
        "--report_path",
        type=str,
        help="Path to export reports details related with results of continuous experimentation." ,
        default=None,
    )

    parser.add_argument(
        "--report_name",
        type=str,
        help="Report name to save reports details related with results of continuous experimentation." ,
        default=None,
    )

    return parser.parse_args(args)

def main(args: Optional[List[Any]] = None) -> None:
    """Run the `better_experimentation` package.

    Args:
      args: Arguments for the programme (Default value=None).
    """

    # Parse the arguments
    parsed_args = parse_args(args)
    kwargs = vars(parsed_args)

    # Generate the profiling report
    better_exp = BetterExperimentation(
        models_trained=kwargs["models_trained_path"],
        X_test=kwargs["x_test_path"],
        y_test=kwargs["y_test_path"],
        return_best_model=True,
        **kwargs
    )
    best_model = better_exp.run()
    return best_model