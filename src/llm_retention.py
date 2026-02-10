"""
LLM-Powered Retention Message Generator
Generates personalized retention messages for high-risk customers.
Uses Claude API (Anthropic) or falls back to template-based generation.
"""

import os
import json
from typing import Optional


def generate_retention_message_template(customer_profile: dict, churn_probability: float, shap_drivers: list) -> str:
    """
    Template-based retention message generator (no API needed).
    Uses customer profile and SHAP drivers to craft personalized messages.
    """
    name = customer_profile.get("customer_id", "Valued Customer")
    tenure = customer_profile.get("tenure", 0)
    contract = customer_profile.get("contract", "Month-to-month")
    monthly_charges = customer_profile.get("monthly_charges", 0)
    internet = customer_profile.get("internet_service", "Unknown")

    risk_level = "critical" if churn_probability > 0.8 else "high" if churn_probability > 0.6 else "moderate"

    # Identify top churn drivers
    top_drivers = []
    for d in shap_drivers[:3]:
        feat = d.get("feature", "")
        if "contract" in feat.lower() or "month-to-month" in feat.lower():
            top_drivers.append("contract_flexibility")
        elif "tenure" in feat.lower() or "new_customer" in feat.lower():
            top_drivers.append("new_customer")
        elif "charge" in feat.lower() or "overpay" in feat.lower():
            top_drivers.append("pricing")
        elif "security" in feat.lower() or "support" in feat.lower() or "tech" in feat.lower():
            top_drivers.append("service_gap")
        elif "fiber" in feat.lower() or "internet" in feat.lower():
            top_drivers.append("service_quality")
        elif "paperless" in feat.lower() or "electronic" in feat.lower():
            top_drivers.append("billing")
        else:
            top_drivers.append("general")

    # Build personalized message components
    greeting = f"Dear Customer #{name},"
    
    # Tenure-based opening
    if tenure <= 6:
        opening = "Welcome to our family! We noticed you're still getting started with us, and we want to make sure you're getting the most out of your services."
    elif tenure <= 24:
        opening = f"Thank you for being with us for {tenure} months. Your loyalty means a lot, and we want to ensure you continue to have an excellent experience."
    else:
        opening = f"As a valued long-term customer of {tenure} months, you're incredibly important to us. We'd like to show our appreciation."

    # Driver-specific offers
    offers = []
    if "contract_flexibility" in top_drivers:
        offers.append("🔒 **Exclusive Offer**: Switch to an annual plan and save 20% on your monthly charges — that's over ${:.0f}/year in savings!".format(monthly_charges * 0.2 * 12))
    if "pricing" in top_drivers or "billing" in top_drivers:
        discount = monthly_charges * 0.15
        offers.append(f"💰 **Price Lock Guarantee**: We're offering you a ${discount:.0f}/month discount for the next 6 months as a loyalty reward.")
    if "service_gap" in top_drivers:
        offers.append("🛡️ **Free Upgrade**: Get our Premium Security & Tech Support bundle FREE for 3 months — protecting your devices and giving you 24/7 expert support.")
    if "new_customer" in top_drivers:
        offers.append("🎁 **New Customer Bonus**: Enjoy a complimentary month of our StreamingTV + StreamingMovies bundle to explore all we have to offer.")
    if "service_quality" in top_drivers:
        offers.append("⚡ **Speed Boost**: We're upgrading your internet speed tier at no additional cost for the next 6 months.")
    
    if not offers:
        offers.append(f"🌟 **Loyalty Reward**: As a thank you, we're crediting ${monthly_charges * 0.1:.0f} to your next bill.")

    # Risk-based urgency
    if risk_level == "critical":
        urgency = "This personalized offer is available for the next 48 hours. Don't miss out!"
    elif risk_level == "high":
        urgency = "This exclusive offer is reserved just for you and expires in 7 days."
    else:
        urgency = "Take advantage of this offer at your convenience — it's our way of saying thanks."

    # Assemble message
    message = f"""{greeting}

{opening}

We've put together something special just for you:

{chr(10).join(offers)}

{urgency}

To claim your offer, simply reply to this message or call us at 1-800-TELCO. Our team is ready to help you get set up.

We truly value your business and look forward to serving you for years to come.

Warm regards,
The Customer Success Team
Telco Revenue Intelligence"""

    return message


def generate_retention_message_llm(
    customer_profile: dict,
    churn_probability: float,
    shap_drivers: list,
    api_key: Optional[str] = None,
) -> str:
    """
    Use Claude API to generate a personalized retention message.
    Falls back to template if no API key is available.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return generate_retention_message_template(customer_profile, churn_probability, shap_drivers)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""You are a customer retention specialist for a telecom company. 
Generate a personalized, empathetic retention email for this high-risk customer.

Customer Profile:
- Customer ID: {customer_profile.get('customer_id', 'N/A')}
- Tenure: {customer_profile.get('tenure', 0)} months
- Contract: {customer_profile.get('contract', 'N/A')}
- Monthly Charges: ${customer_profile.get('monthly_charges', 0):.2f}
- Internet Service: {customer_profile.get('internet_service', 'N/A')}
- Churn Probability: {churn_probability:.1%}

Top Churn Risk Drivers (from SHAP analysis):
{json.dumps(shap_drivers[:5], indent=2)}

Requirements:
1. Be warm, personal, and empathetic
2. Address the specific risk factors identified by SHAP
3. Include a concrete, quantified offer (discount, upgrade, or bonus)
4. Create appropriate urgency without being pushy
5. Keep it under 200 words
6. Include a clear call-to-action"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as e:
        print(f"⚠️ LLM API failed ({e}), falling back to template.")
        return generate_retention_message_template(customer_profile, churn_probability, shap_drivers)


def batch_generate_messages(
    high_risk_customers: list,
    use_llm: bool = False,
    api_key: Optional[str] = None,
) -> list:
    """Generate retention messages for a batch of high-risk customers."""
    messages = []
    for customer in high_risk_customers:
        if use_llm:
            msg = generate_retention_message_llm(
                customer["profile"], customer["churn_prob"], customer["shap_drivers"], api_key
            )
        else:
            msg = generate_retention_message_template(
                customer["profile"], customer["churn_prob"], customer["shap_drivers"]
            )
        messages.append({
            "customer_id": customer["profile"].get("customer_id"),
            "churn_probability": customer["churn_prob"],
            "message": msg,
        })
    return messages


if __name__ == "__main__":
    # Demo
    sample_profile = {
        "customer_id": "7590-VHVEG",
        "tenure": 3,
        "contract": "Month-to-month",
        "monthly_charges": 89.50,
        "internet_service": "Fiber optic",
    }
    sample_drivers = [
        {"feature": "Contract_Month-to-month", "shap_impact": 0.45, "direction": "increases"},
        {"feature": "tenure", "shap_impact": -0.32, "direction": "increases"},
        {"feature": "MonthlyCharges", "shap_impact": 0.28, "direction": "increases"},
    ]
    print(generate_retention_message_template(sample_profile, 0.85, sample_drivers))
