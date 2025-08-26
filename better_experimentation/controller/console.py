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

    parser.add_argument(
        '--test_data_paths',
        nargs='+',
        help='Tuples contains test data path to experiment, considering in this order: X_test path, y_test path, name of the test data. You can define more than one tuple of three values'
    )

    parser.add_argument(
        '--contexts',
        nargs='+',
        help='Tuples of samples to experiment, considering in this order: model path, test data name, name to reference the pair of sample (model+test_data). You can define more than one tuple of three values'
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

    test_data_paths = kwargs["test_data_paths"]
    del kwargs["test_data_paths"]

    contexts = kwargs["contexts"]
    del kwargs["contexts"]


    # Generate the profiling report
    better_exp = BetterExperimentation(
        return_best_model=True,
        **kwargs
    )

    for i in range(0, len(test_data_paths), 3):
        better_exp.add_test_data(test_data_name=test_data_paths[i+2],
                                 X_test=test_data_paths[i],
                                 y_test=test_data_paths[i+1])

    for i in range(0, len(contexts), 3):
        print(contexts[i], contexts[i+1], contexts[i+2])
        better_exp.add_context(ref_test_data=contexts[i+1],
                             context_name=contexts[i+2],
                             model_trained=contexts[i])

    best_model = better_exp.run()
    return best_model