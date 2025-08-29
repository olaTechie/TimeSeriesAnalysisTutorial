"""
Report Generation Module for Air Quality Analysis
This module generates comprehensive analysis reports in various formats.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, simulation_results: Optional[Dict] = None,
                 spatial_analysis: Optional[pd.DataFrame] = None,
                 model_results: Optional[Dict] = None):
        """
        Initialize report generator
        
        Args:
            simulation_results: Results from intervention simulations
            spatial_analysis: Spatial analysis results
            model_results: Model performance results
        """
        self.simulation_results = simulation_results or {}
        self.spatial_analysis = spatial_analysis
        self.model_results = model_results or {}
        self.report_date = datetime.now().strftime('%Y-%m-%d')
        logger.info("ReportGenerator initialized")
    
    def generate_executive_summary(self, include_recommendations: bool = True) -> str:
        """
        Generate executive summary of findings
        
        Args:
            include_recommendations: Whether to include recommendations
            
        Returns:
            Executive summary text
        """
        summary = []
        summary.append("=" * 80)
        summary.append("EXECUTIVE SUMMARY: Air Quality Interventions and Respiratory Health Impact")
        summary.append(f"Report Date: {self.report_date}")
        summary.append("=" * 80)
        summary.append("")
        
        # Key findings
        summary.append("KEY FINDINGS:")
        summary.append("-" * 40)
        
        # Best intervention scenario
        if self.simulation_results:
            best_scenario = self._identify_best_scenario()
            if best_scenario:
                summary.append(f"• Most effective intervention: {best_scenario['name']}")
                summary.append(f"  - PM2.5 reduction potential: {best_scenario['pm25_reduction']:.1f}%")
                summary.append(f"  - Estimated cases prevented: {best_scenario['cases_prevented']:.0f}")
                summary.append(f"  - Cost factor: {best_scenario['cost_factor']:.1f}x baseline")
        
        # Spatial findings
        if self.spatial_analysis is not None:
            summary.append("")
            summary.append("SPATIAL ANALYSIS:")
            summary.append("-" * 40)
            
            if 'vulnerability_index' in self.spatial_analysis.columns:
                highest_vuln = self.spatial_analysis['vulnerability_index'].idxmax()
                lowest_vuln = self.spatial_analysis['vulnerability_index'].idxmin()
                summary.append(f"• Highest vulnerability county: {highest_vuln}")
                summary.append(f"• Lowest vulnerability county: {lowest_vuln}")
            
            if 'avg_pm2_5_calibrated_mean' in self.spatial_analysis.columns:
                avg_pm25 = self.spatial_analysis['avg_pm2_5_calibrated_mean'].mean()
                max_pm25 = self.spatial_analysis['avg_pm2_5_calibrated_mean'].max()
                summary.append(f"• Average PM2.5 across counties: {avg_pm25:.2f} μg/m³")
                summary.append(f"• Maximum county average: {max_pm25:.2f} μg/m³")
        
        # Model performance
        if self.model_results:
            summary.append("")
            summary.append("PREDICTIVE MODEL PERFORMANCE:")
            summary.append("-" * 40)
            
            best_model = self._identify_best_model()
            if best_model:
                summary.append(f"• Best performing model: {best_model['name']}")
                summary.append(f"  - Accuracy: {best_model['accuracy']:.3f}")
                summary.append(f"  - F1 Score: {best_model['f1_score']:.3f}")
        
        # Recommendations
        if include_recommendations:
            summary.append("")
            summary.append("RECOMMENDATIONS:")
            summary.append("-" * 40)
            recommendations = self._generate_recommendations()
            for rec in recommendations:
                summary.append(f"• {rec}")
        
        summary.append("")
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def generate_technical_report(self, include_methodology: bool = True,
                                 include_limitations: bool = True) -> str:
        """
        Generate detailed technical report
        
        Args:
            include_methodology: Whether to include methodology section
            include_limitations: Whether to include limitations section
            
        Returns:
            Technical report text
        """
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("DETAILED TECHNICAL REPORT")
        report.append("Simulation-Based Epidemiological Analysis of Air Quality Interventions")
        report.append(f"Generated: {self.report_date}")
        report.append("=" * 80)
        report.append("")
        
        # Table of Contents
        report.append("TABLE OF CONTENTS")
        report.append("-" * 40)
        report.append("1. Executive Summary")
        report.append("2. Methodology")
        report.append("3. Data Overview")
        report.append("4. Intervention Scenarios")
        report.append("5. Health Impact Assessment")
        report.append("6. Spatial Analysis")
        report.append("7. Predictive Modeling")
        report.append("8. Cost-Effectiveness Analysis")
        report.append("9. Limitations")
        report.append("10. Conclusions and Recommendations")
        report.append("")
        
        # 1. Executive Summary (brief version)
        report.append("1. EXECUTIVE SUMMARY")
        report.append("-" * 40)
        exec_summary = self._generate_brief_summary()
        report.append(exec_summary)
        report.append("")
        
        # 2. Methodology
        if include_methodology:
            report.append("2. METHODOLOGY")
            report.append("-" * 40)
            report.append("• Data Integration: Combined environmental (PM2.5) and health surveillance data")
            report.append("• Temporal Resolution: Monthly aggregation from 2020-2024")
            report.append("• Spatial Resolution: County-level analysis")
            report.append("• Modeling Approach:")
            report.append("  - Intervention simulation using sigmoid implementation curves")
            report.append("  - Concentration-response functions for health impact estimation")
            report.append("  - Time series classification for predictive modeling")
            report.append("  - Spatial analysis for vulnerability assessment")
            report.append("")
        
        # 3. Data Overview
        report.append("3. DATA OVERVIEW")
        report.append("-" * 40)
        data_summary = self._generate_data_summary()
        report.append(data_summary)
        report.append("")
        
        # 4. Intervention Scenarios
        report.append("4. INTERVENTION SCENARIOS ANALYZED")
        report.append("-" * 40)
        
        if self.simulation_results:
            for scenario_name, results in self.simulation_results.items():
                scenario = results['scenario']
                report.append(f"\n{scenario.name}:")
                report.append(f"  - PM2.5 Reduction Potential: {scenario.pm25_reduction*100:.1f}%")
                report.append(f"  - Implementation Time: {scenario.implementation_time} months")
                report.append(f"  - Cost Factor: {scenario.cost_factor:.1f}x baseline")
                report.append(f"  - Sustainability Score: {scenario.sustainability_score:.2f}/1.0")
                
                if 'summary_stats' in results:
                    stats = results['summary_stats']
                    report.append(f"  - Total Cases Prevented: {stats.get('total_cases_prevented', 0):.0f}")
        report.append("")
        
        # 5. Health Impact Assessment
        report.append("5. HEALTH IMPACT ASSESSMENT")
        report.append("-" * 40)
        health_assessment = self._generate_health_assessment()
        report.append(health_assessment)
        report.append("")
        
        # 6. Spatial Analysis
        report.append("6. SPATIAL ANALYSIS")
        report.append("-" * 40)
        spatial_summary = self._generate_spatial_summary()
        report.append(spatial_summary)
        report.append("")
        
        # 7. Predictive Modeling
        report.append("7. PREDICTIVE MODELING RESULTS")
        report.append("-" * 40)
        model_summary = self._generate_model_summary()
        report.append(model_summary)
        report.append("")
        
        # 8. Cost-Effectiveness
        report.append("8. COST-EFFECTIVENESS ANALYSIS")
        report.append("-" * 40)
        cost_analysis = self._generate_cost_analysis()
        report.append(cost_analysis)
        report.append("")
        
        # 9. Limitations
        if include_limitations:
            report.append("9. LIMITATIONS")
            report.append("-" * 40)
            report.append("• Data Limitations:")
            report.append("  - PM2.5 monitoring limited to specific sites")
            report.append("  - Health data aggregated at monthly level")
            report.append("  - Potential underreporting of respiratory cases")
            report.append("• Modeling Assumptions:")
            report.append("  - Linear concentration-response relationships")
            report.append("  - Uniform implementation across counties")
            report.append("  - No interaction effects between interventions")
            report.append("• Temporal Constraints:")
            report.append("  - Limited historical data period")
            report.append("  - Seasonal variations may not be fully captured")
            report.append("")
        
        # 10. Conclusions and Recommendations
        report.append("10. CONCLUSIONS AND RECOMMENDATIONS")
        report.append("-" * 40)
        conclusions = self._generate_conclusions()
        report.append(conclusions)
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_markdown_report(self) -> str:
        """
        Generate report in Markdown format
        
        Returns:
            Markdown formatted report
        """
        md = []
        
        # Header
        md.append("# Air Quality Intervention Analysis Report")
        md.append(f"*Generated: {self.report_date}*")
        md.append("")
        
        # Executive Summary
        md.append("## Executive Summary")
        md.append("")
        
        if self.simulation_results:
            best_scenario = self._identify_best_scenario()
            if best_scenario:
                md.append(f"**Most Effective Intervention:** {best_scenario['name']}")
                md.append("")
                md.append("| Metric | Value |")
                md.append("|--------|-------|")
                md.append(f"| PM2.5 Reduction | {best_scenario['pm25_reduction']:.1f}% |")
                md.append(f"| Cases Prevented | {best_scenario['cases_prevented']:.0f} |")
                md.append(f"| Cost Factor | {best_scenario['cost_factor']:.1f}x |")
                md.append("")
        
        # Key Findings
        md.append("## Key Findings")
        md.append("")
        findings = self._generate_key_findings()
        for finding in findings:
            md.append(f"- {finding}")
        md.append("")
        
        # Intervention Comparison Table
        md.append("## Intervention Scenarios")
        md.append("")
        
        if self.simulation_results:
            md.append("| Scenario | PM2.5 Reduction | Implementation Time | Cost Factor | Sustainability |")
            md.append("|----------|----------------|-------------------|-------------|----------------|")
            
            for name, results in self.simulation_results.items():
                scenario = results['scenario']
                md.append(f"| {scenario.name} | {scenario.pm25_reduction*100:.1f}% | "
                         f"{scenario.implementation_time} months | {scenario.cost_factor:.1f}x | "
                         f"{scenario.sustainability_score:.2f} |")
            md.append("")
        
        # Spatial Analysis
        if self.spatial_analysis is not None:
            md.append("## Spatial Analysis")
            md.append("")
            md.append("### County Vulnerability Ranking")
            md.append("")
            
            if 'vulnerability_index' in self.spatial_analysis.columns:
                sorted_counties = self.spatial_analysis.sort_values('vulnerability_index', ascending=False)
                md.append("| Rank | County | Vulnerability Index | Avg PM2.5 (μg/m³) |")
                md.append("|------|--------|-------------------|------------------|")
                
                for i, (county, row) in enumerate(sorted_counties.head(5).iterrows(), 1):
                    pm25 = row.get('avg_pm2_5_calibrated_mean', 0)
                    vuln = row.get('vulnerability_index', 0)
                    md.append(f"| {i} | {county} | {vuln:.3f} | {pm25:.1f} |")
                md.append("")
        
        # Model Performance
        if self.model_results:
            md.append("## Predictive Model Performance")
            md.append("")
            md.append("| Model | Accuracy | F1 Score | Precision | Recall |")
            md.append("|-------|----------|----------|-----------|--------|")
            
            for model_name, results in self.model_results.items():
                md.append(f"| {model_name} | {results.get('avg_accuracy', 0):.3f} | "
                         f"{results.get('avg_f1', 0):.3f} | {results.get('avg_precision', 0):.3f} | "
                         f"{results.get('avg_recall', 0):.3f} |")
            md.append("")
        
        # Recommendations
        md.append("## Recommendations")
        md.append("")
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            md.append(f"{i}. {rec}")
        md.append("")
        
        return "\n".join(md)
    
    def generate_html_report(self) -> str:
        """
        Generate report in HTML format
        
        Returns:
            HTML formatted report
        """
        html = []
        
        # HTML header
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Air Quality Intervention Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th { background-color: #3498db; color: white; padding: 12px; text-align: left; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background-color: #f5f5f5; }
                .summary-box { background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .metric { display: inline-block; margin: 10px 20px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .metric-label { color: #7f8c8d; font-size: 14px; }
                .recommendation { background-color: #e8f6f3; padding: 10px; margin: 10px 0; border-left: 4px solid #27ae60; }
            </style>
        </head>
        <body>
        """)
        
        # Report header
        html.append(f"<h1>Air Quality Intervention Analysis Report</h1>")
        html.append(f"<p><em>Generated: {self.report_date}</em></p>")
        
        # Executive Summary
        html.append("<div class='summary-box'>")
        html.append("<h2>Executive Summary</h2>")
        
        if self.simulation_results:
            best_scenario = self._identify_best_scenario()
            if best_scenario:
                html.append("<div class='metrics'>")
                html.append(f"<div class='metric'><div class='metric-label'>Best Intervention</div>"
                           f"<div class='metric-value'>{best_scenario['name']}</div></div>")
                html.append(f"<div class='metric'><div class='metric-label'>PM2.5 Reduction</div>"
                           f"<div class='metric-value'>{best_scenario['pm25_reduction']:.1f}%</div></div>")
                html.append(f"<div class='metric'><div class='metric-label'>Cases Prevented</div>"
                           f"<div class='metric-value'>{best_scenario['cases_prevented']:.0f}</div></div>")
                html.append("</div>")
        
        html.append("</div>")
        
        # Intervention Scenarios Table
        html.append("<h2>Intervention Scenarios</h2>")
        html.append("<table>")
        html.append("<tr><th>Scenario</th><th>PM2.5 Reduction</th><th>Implementation Time</th>"
                   "<th>Cost Factor</th><th>Sustainability</th></tr>")
        
        if self.simulation_results:
            for name, results in self.simulation_results.items():
                scenario = results['scenario']
                html.append(f"<tr><td>{scenario.name}</td><td>{scenario.pm25_reduction*100:.1f}%</td>"
                           f"<td>{scenario.implementation_time} months</td><td>{scenario.cost_factor:.1f}x</td>"
                           f"<td>{scenario.sustainability_score:.2f}</td></tr>")
        
        html.append("</table>")
        
        # Recommendations
        html.append("<h2>Recommendations</h2>")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            html.append(f"<div class='recommendation'>{rec}</div>")
        
        # HTML footer
        html.append("</body></html>")
        
        return "\n".join(html)
    
    def _identify_best_scenario(self) -> Optional[Dict]:
        """Identify the best performing scenario"""
        if not self.simulation_results:
            return None
        
        best_scenario = None
        max_benefit = 0
        
        for name, results in self.simulation_results.items():
            if name == 'baseline':
                continue
            
            scenario = results['scenario']
            stats = results.get('summary_stats', {})
            
            # Calculate benefit score (cases prevented / cost factor)
            cases_prevented = stats.get('total_cases_prevented', 0)
            benefit_score = cases_prevented / scenario.cost_factor if scenario.cost_factor > 0 else 0
            
            if benefit_score > max_benefit:
                max_benefit = benefit_score
                best_scenario = {
                    'name': scenario.name,
                    'pm25_reduction': scenario.pm25_reduction * 100,
                    'cases_prevented': cases_prevented,
                    'cost_factor': scenario.cost_factor,
                    'benefit_score': benefit_score
                }
        
        return best_scenario
    
    def _identify_best_model(self) -> Optional[Dict]:
        """Identify the best performing model"""
        if not self.model_results:
            return None
        
        best_model = None
        best_f1 = 0
        
        for model_name, results in self.model_results.items():
            f1_score = results.get('avg_f1', 0)
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = {
                    'name': model_name,
                    'accuracy': results.get('avg_accuracy', 0),
                    'f1_score': f1_score,
                    'precision': results.get('avg_precision', 0),
                    'recall': results.get('avg_recall', 0)
                }
        
        return best_model
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on best scenario
        best_scenario = self._identify_best_scenario()
        if best_scenario:
            if 'traffic' in best_scenario['name'].lower():
                recommendations.append("Prioritize traffic emission control measures for quick impact")
            elif 'green' in best_scenario['name'].lower():
                recommendations.append("Invest in urban green space expansion for sustainable long-term benefits")
            elif 'combined' in best_scenario['name'].lower():
                recommendations.append("Implement combined interventions for maximum health impact")
        
        # Based on spatial analysis
        if self.spatial_analysis is not None and 'vulnerability_index' in self.spatial_analysis.columns:
            high_vuln = self.spatial_analysis['vulnerability_index'].idxmax()
            recommendations.append(f"Focus initial interventions in {high_vuln} county due to high vulnerability")
        
        # Based on model performance
        if self.model_results:
            best_model = self._identify_best_model()
            if best_model and best_model['accuracy'] > 0.8:
                recommendations.append("Deploy predictive models for early warning systems")
        
        # General recommendations
        recommendations.extend([
            "Establish continuous air quality monitoring in all counties",
            "Develop county-specific intervention strategies based on local conditions",
            "Create public awareness campaigns about air quality and health impacts",
            "Implement regular evaluation of intervention effectiveness"
        ])
        
        return recommendations[:6]  # Return top 6 recommendations
    
    def _generate_brief_summary(self) -> str:
        """Generate brief summary for technical report"""
        summary = []
        
        if self.simulation_results:
            n_scenarios = len(self.simulation_results)
            summary.append(f"Analyzed {n_scenarios} intervention scenarios for air quality improvement.")
        
        best_scenario = self._identify_best_scenario()
        if best_scenario:
            summary.append(f"The most effective intervention ({best_scenario['name']}) "
                          f"could reduce PM2.5 by {best_scenario['pm25_reduction']:.1f}% "
                          f"and prevent approximately {best_scenario['cases_prevented']:.0f} "
                          f"respiratory disease cases.")
        
        if self.spatial_analysis is not None:
            n_counties = len(self.spatial_analysis)
            summary.append(f"Spatial analysis covered {n_counties} counties with varying "
                          f"vulnerability levels and exposure patterns.")
        
        return " ".join(summary)
    
    def _generate_data_summary(self) -> str:
        """Generate data overview summary"""
        summary = []
        
        summary.append("Data Sources:")
        summary.append("• PM2.5 exposure data from air quality monitoring stations")
        summary.append("• Health outcome data from DHIS2 surveillance system")
        summary.append("• Monthly aggregation from 2020-2024")
        
        if self.spatial_analysis is not None:
            n_counties = len(self.spatial_analysis)
            summary.append(f"• Geographic coverage: {n_counties} counties")
        
        return "\n".join(summary)
    
    def _generate_health_assessment(self) -> str:
        """Generate health impact assessment summary"""
        summary = []
        
        summary.append("Concentration-Response Functions Applied:")
        summary.append("• Pneumonia: 12% increase per 10 μg/m³ PM2.5")
        summary.append("• Tuberculosis: 8% increase per 10 μg/m³ PM2.5")
        summary.append("• ILI: 15% increase per 10 μg/m³ PM2.5")
        summary.append("• SARI: 20% increase per 10 μg/m³ PM2.5")
        
        if self.simulation_results:
            total_prevented = 0
            for name, results in self.simulation_results.items():
                if 'summary_stats' in results:
                    total_prevented += results['summary_stats'].get('total_cases_prevented', 0)
            
            summary.append(f"\nTotal potential cases prevented across all scenarios: {total_prevented:.0f}")
        
        return "\n".join(summary)
    
    def _generate_spatial_summary(self) -> str:
        """Generate spatial analysis summary"""
        if self.spatial_analysis is None:
            return "Spatial analysis data not available."
        
        summary = []
        
        if 'avg_pm2_5_calibrated_mean' in self.spatial_analysis.columns:
            mean_pm25 = self.spatial_analysis['avg_pm2_5_calibrated_mean'].mean()
            std_pm25 = self.spatial_analysis['avg_pm2_5_calibrated_mean'].std()
            summary.append(f"PM2.5 Distribution: {mean_pm25:.2f} ± {std_pm25:.2f} μg/m³")
        
        if 'vulnerability_index' in self.spatial_analysis.columns:
            high_vuln = (self.spatial_analysis['vulnerability_index'] > 0.7).sum()
            summary.append(f"High vulnerability counties: {high_vuln}")
        
        return "\n".join(summary)
    
    def _generate_model_summary(self) -> str:
        """Generate model performance summary"""
        if not self.model_results:
            return "Model results not available."
        
        summary = []
        
        best_model = self._identify_best_model()
        if best_model:
            summary.append(f"Best Model: {best_model['name']}")
            summary.append(f"• Accuracy: {best_model['accuracy']:.3f}")
            summary.append(f"• F1 Score: {best_model['f1_score']:.3f}")
        
        summary.append(f"\nTotal models evaluated: {len(self.model_results)}")
        
        return "\n".join(summary)
    
    def _generate_cost_analysis(self) -> str:
        """Generate cost-effectiveness analysis"""
        summary = []
        
        if self.simulation_results:
            cost_effective = []
            
            for name, results in self.simulation_results.items():
                if name == 'baseline':
                    continue
                
                scenario = results['scenario']
                stats = results.get('summary_stats', {})
                cases_prevented = stats.get('total_cases_prevented', 0)
                
                if cases_prevented > 0:
                    cost_per_case = scenario.cost_factor / cases_prevented * 1000  # Arbitrary scale
                    cost_effective.append((scenario.name, cost_per_case))
            
            if cost_effective:
                cost_effective.sort(key=lambda x: x[1])
                summary.append("Cost-Effectiveness Ranking (cost per case prevented):")
                for i, (name, cost) in enumerate(cost_effective[:3], 1):
                    summary.append(f"{i}. {name}: ${cost:.2f}")
        
        return "\n".join(summary) if summary else "Cost analysis not available."
    
    def _generate_conclusions(self) -> str:
        """Generate conclusions section"""
        conclusions = []
        
        conclusions.append("This analysis demonstrates that targeted air quality interventions "
                          "can significantly reduce PM2.5 exposure and prevent respiratory disease cases.")
        
        best_scenario = self._identify_best_scenario()
        if best_scenario:
            conclusions.append(f"\nThe {best_scenario['name']} intervention shows the most promise "
                              f"with a potential {best_scenario['pm25_reduction']:.1f}% reduction in PM2.5.")
        
        conclusions.append("\nImplementation should prioritize high-vulnerability counties "
                          "and consider local conditions for maximum effectiveness.")
        
        return "\n".join(conclusions)
    
    def _generate_key_findings(self) -> List[str]:
        """Generate list of key findings"""
        findings = []
        
        if self.simulation_results:
            findings.append(f"Evaluated {len(self.simulation_results)} intervention scenarios")
        
        best_scenario = self._identify_best_scenario()
        if best_scenario:
            findings.append(f"Top intervention can reduce PM2.5 by {best_scenario['pm25_reduction']:.1f}%")
        
        if self.spatial_analysis is not None:
            findings.append(f"Identified {len(self.spatial_analysis)} counties with varying vulnerability")
        
        if self.model_results:
            findings.append(f"Trained {len(self.model_results)} predictive models")
        
        return findings
    
    def save_reports(self, output_dir: str = './reports'):
        """
        Save all report formats to files
        
        Args:
            output_dir: Directory to save reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save executive summary
        exec_summary = self.generate_executive_summary()
        with open(output_path / 'executive_summary.txt', 'w') as f:
            f.write(exec_summary)
        logger.info(f"Saved executive summary to {output_path / 'executive_summary.txt'}")
        
        # Save technical report
        tech_report = self.generate_technical_report()
        with open(output_path / 'technical_report.txt', 'w') as f:
            f.write(tech_report)
        logger.info(f"Saved technical report to {output_path / 'technical_report.txt'}")
        
        # Save markdown report
        md_report = self.generate_markdown_report()
        with open(output_path / 'report.md', 'w') as f:
            f.write(md_report)
        logger.info(f"Saved markdown report to {output_path / 'report.md'}")
        
        # Save HTML report
        html_report = self.generate_html_report()
        with open(output_path / 'report.html', 'w') as f:
            f.write(html_report)
        logger.info(f"Saved HTML report to {output_path / 'report.html'}")