"""Report generation module for kalmanbox.

Provides automated report generation for state-space model results
in HTML, LaTeX, and Markdown formats.

Usage
-----
>>> from kalmanbox.reports import ReportManager
>>> rm = ReportManager()
>>> rm.generate(results, report_type='ssm', fmt='html', output='report.html')
"""

from kalmanbox.reports.report_manager import ReportManager

__all__ = ["ReportManager"]
