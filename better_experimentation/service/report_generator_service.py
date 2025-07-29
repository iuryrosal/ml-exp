from jinja2 import Environment, PackageLoader, select_autoescape
import json
from importlib import resources

from better_experimentation.utils.log_config import LogService, handle_exceptions


class ReportGeneratorService:
    """Responsible to generate HTML report with all details related with results from experimental pipeline applied in models
    """
    __log_service = LogService()
    def __init__(self, reports, report_name, report_base_path) -> None:
        self.__logger = self.__log_service.get_logger(__name__)

        self.env = Environment(
            loader=PackageLoader("better_experimentation", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = self.env.get_template("report.html")
        results_data = json.loads(reports.json())

        html_renderizado = template.render(reports_by_score=results_data["reports_by_score"],
                                           message_about_significancy=results_data["message_about_significancy"],
                                           better_model_by_score=results_data["better_model_by_score"])

        with open(f"{report_base_path}/{report_name}.html", "w") as f:
            f.write(html_renderizado)