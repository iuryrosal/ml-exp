from jinja2 import Environment, FileSystemLoader
import json


class ReportGeneratorService:
    def __init__(self, reports) -> None:
        env = Environment(loader=FileSystemLoader("ml_continuous_test/templates"))
        template = env.get_template("report.html")
        results_data = json.loads(reports.json())

        html_renderizado = template.render(reports_by_score=results_data["reports_by_score"],
                                           message_about_significancy=results_data["message_about_significancy"],
                                           better_model_by_score=results_data["better_model_by_score"])

        with open("reports/relatorio_final.html", "w") as f:
            f.write(html_renderizado)