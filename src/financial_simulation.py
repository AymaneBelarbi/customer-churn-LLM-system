"""
Financial Impact Simulation
Simulates revenue impact of reducing churn by X%.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np


def simulate_financial_impact(
    data_path: str = "data/processed/featured.csv",
    churn_reduction_pct: float = 5.0,
):
    """
    Simulate financial impact of reducing churn.

    Assumptions:
    - Each churned customer's MonthlyCharges is lost revenue per month
    - Average customer lifetime after retention = 12 months
    - Cost of retention campaign per customer = $50
    - Discount offered to retain = 10% of MonthlyCharges for 3 months
    """
    df = pd.read_csv(data_path)

    total_customers = len(df)
    churned = df[df["Churn"] == 1]
    n_churned = len(churned)
    churn_rate = n_churned / total_customers

    # Current metrics
    avg_monthly_revenue = df["MonthlyCharges"].mean()
    total_monthly_revenue = df["MonthlyCharges"].sum()
    churned_monthly_revenue = churned["MonthlyCharges"].sum()
    annual_revenue_loss = churned_monthly_revenue * 12

    # Reduction scenario
    reduction_fraction = churn_reduction_pct / 100
    customers_saved = int(n_churned * reduction_fraction)
    saved_customers_df = churned.sample(n=customers_saved, random_state=42)

    # Revenue recovered
    monthly_revenue_recovered = saved_customers_df["MonthlyCharges"].sum()
    annual_revenue_recovered = monthly_revenue_recovered * 12

    # Costs
    retention_campaign_cost_per_customer = 50
    discount_pct = 0.10
    discount_months = 3
    total_campaign_cost = customers_saved * retention_campaign_cost_per_customer
    total_discount_cost = (saved_customers_df["MonthlyCharges"] * discount_pct * discount_months).sum()
    total_retention_cost = total_campaign_cost + total_discount_cost

    # Net impact
    net_annual_impact = annual_revenue_recovered - total_retention_cost
    roi = (net_annual_impact / total_retention_cost) * 100 if total_retention_cost > 0 else 0

    # New churn rate
    new_churn_rate = (n_churned - customers_saved) / total_customers

    results = {
        "scenario": f"Reduce churn by {churn_reduction_pct}%",
        "current_metrics": {
            "total_customers": total_customers,
            "churned_customers": n_churned,
            "churn_rate": round(churn_rate * 100, 2),
            "avg_monthly_revenue_per_customer": round(avg_monthly_revenue, 2),
            "total_monthly_revenue": round(total_monthly_revenue, 2),
            "annual_revenue_loss_from_churn": round(annual_revenue_loss, 2),
        },
        "intervention": {
            "churn_reduction_target": f"{churn_reduction_pct}%",
            "customers_saved": customers_saved,
            "new_churn_rate": round(new_churn_rate * 100, 2),
        },
        "financial_impact": {
            "monthly_revenue_recovered": round(monthly_revenue_recovered, 2),
            "annual_revenue_recovered": round(annual_revenue_recovered, 2),
            "retention_campaign_cost": round(total_campaign_cost, 2),
            "discount_cost": round(total_discount_cost, 2),
            "total_retention_cost": round(total_retention_cost, 2),
            "net_annual_impact": round(net_annual_impact, 2),
            "roi_pct": round(roi, 1),
        },
        "sensitivity_analysis": [],
    }

    # Sensitivity: 1% to 15% reduction
    for pct in [1, 2, 3, 5, 7, 10, 15]:
        frac = pct / 100
        saved = int(n_churned * frac)
        saved_df = churned.sample(n=max(saved, 1), random_state=42)
        recovered = saved_df["MonthlyCharges"].sum() * 12
        cost = saved * 50 + (saved_df["MonthlyCharges"] * 0.10 * 3).sum()
        net = recovered - cost
        results["sensitivity_analysis"].append({
            "reduction_pct": pct,
            "customers_saved": saved,
            "annual_revenue_recovered": round(recovered, 2),
            "total_cost": round(cost, 2),
            "net_impact": round(net, 2),
            "roi_pct": round((net / cost) * 100, 1) if cost > 0 else 0,
        })

    # Save results
    out_dir = Path("static")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "financial_simulation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("💰 FINANCIAL IMPACT SIMULATION")
    print("=" * 60)
    print(f"Current churn rate: {results['current_metrics']['churn_rate']}%")
    print(f"Annual revenue loss from churn: ${annual_revenue_loss:,.0f}")
    print(f"\nScenario: Reduce churn by {churn_reduction_pct}%")
    print(f"Customers saved: {customers_saved}")
    print(f"Annual revenue recovered: ${annual_revenue_recovered:,.0f}")
    print(f"Total retention cost: ${total_retention_cost:,.0f}")
    print(f"Net annual impact: ${net_annual_impact:,.0f}")
    print(f"ROI: {roi:.0f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    simulate_financial_impact(churn_reduction_pct=5.0)
