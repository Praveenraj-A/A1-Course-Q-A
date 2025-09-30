import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import numpy as np

def generate_metrics_report():
    """Generate comprehensive metrics report"""
    
    # Load metrics data (you would load from your actual data store)
    # For demo purposes, we'll create sample data structure
    
    report_data = {
        "summary": {
            "total_queries": 150,
            "time_period": "24 hours",
            "success_rate": 0.89,
            "avg_response_time": 2450,
            "total_cost_estimate": 0.45
        },
        "performance_metrics": {
            "response_time_distribution": [1200, 1800, 2200, 2800, 3500],
            "confidence_scores": [0.7, 0.8, 0.9, 0.6, 0.85],
            "queries_per_hour": [5, 8, 12, 7, 10, 15, 9, 11],
            "error_rates": [0.05, 0.08, 0.03, 0.12, 0.07]
        },
        "cost_analysis": {
            "cost_per_query": 0.003,
            "main_components": ["Gemini API", "Pinecone", "Embedding Generation"],
            "cost_breakdown": [0.002, 0.0005, 0.0005]
        }
    }
    
    # Create visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Response time distribution
    axes[0, 0].hist(report_data["performance_metrics"]["response_time_distribution"], bins=10, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Response Time Distribution (ms)')
    axes[0, 0].set_xlabel('Response Time (ms)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Confidence scores
    axes[0, 1].hist(report_data["performance_metrics"]["confidence_scores"], bins=10, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Confidence Score Distribution')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # Query volume over time
    hours = list(range(len(report_data["performance_metrics"]["queries_per_hour"])))
    axes[1, 0].plot(hours, report_data["performance_metrics"]["queries_per_hour"], marker='o', color='orange')
    axes[1, 0].set_title('Query Volume Over Time')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Queries per Hour')
    
    # Cost breakdown
    components = report_data["cost_analysis"]["main_components"]
    costs = report_data["cost_analysis"]["cost_breakdown"]
    axes[1, 1].pie(costs, labels=components, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Cost Breakdown per Query')
    
    plt.tight_layout()
    plt.savefig('metrics_report.png', dpi=300, bbox_inches='tight')
    
    # Generate CSV report
    csv_data = {
        'timestamp': [datetime.now().isoformat()] * 5,
        'query_example': ['What is machine learning?', 'Explain neural networks', 'Course prerequisites', 'Grading policy', 'Assignment deadlines'],
        'response_time_ms': report_data["performance_metrics"]["response_time_distribution"],
        'confidence_score': report_data["performance_metrics"]["confidence_scores"],
        'relevant_docs_found': [3, 5, 2, 4, 3],
        'success': [True, True, True, False, True]
    }
    
    df = pd.DataFrame(csv_data)
    df.to_csv('performance_metrics.csv', index=False)
    
    # Generate summary report
    summary_report = f"""
    Course Q&A Chatbot - Performance Metrics Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    SUMMARY:
    ---------
    Total Queries Processed: {report_data['summary']['total_queries']}
    Time Period: {report_data['summary']['time_period']}
    Success Rate: {report_data['summary']['success_rate']:.1%}
    Average Response Time: {report_data['summary']['avg_response_time']:.0f} ms
    Total Estimated Cost: ${report_data['summary']['total_cost_estimate']:.2f}
    
    PERFORMANCE INDICATORS:
    ----------------------
    - 95% of queries responded within 4000ms
    - Average confidence score: {np.mean(report_data['performance_metrics']['confidence_scores']):.1%}
    - Peak query volume: {max(report_data['performance_metrics']['queries_per_hour'])} queries/hour
    
    COST ANALYSIS:
    -------------
    Cost per Query: ${report_data['cost_analysis']['cost_per_query']:.3f}
    Most expensive component: {components[costs.index(max(costs))]}
    
    RECOMMENDATIONS:
    ---------------
    1. Optimize retrieval for faster response times
    2. Improve chunking strategy for better relevance
    3. Monitor Gemini API usage for cost optimization
    """
    
    with open('metrics_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print("✅ Metrics report generated:")
    print("   - metrics_report.png (Visualizations)")
    print("   - performance_metrics.csv (Raw data)")
    print("   - metrics_summary.txt (Summary report)")

if __name__ == "__main__":
    generate_metrics_report()