"""
LLM-Powered Retention Message Generator
Generates personalized retention messages for high-risk customers.
Uses a LangChain chain on Claude (Anthropic) with template-based generation as its fallback.
"""

import json
import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENCY = 4

RETENTION_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """You are a customer retention specialist for a telecom company.
Generate a personalized, empathetic retention email for this high-risk customer.

Customer Profile:
- Customer ID: {customer_id}
- Tenure: {tenure} months
- Contract: {contract}
- Monthly Charges: ${monthly_charges}
- Internet Service: {internet_service}
- Churn Probability: {churn_probability}

Top Churn Risk Drivers (from SHAP analysis):
{shap_drivers}

Requirements:
1. Be warm, personal, and empathetic
2. Address the specific risk factors identified by SHAP
3. Include a concrete, quantified offer (discount, upgrade, or bonus)
4. Create appropriate urgency without being pushy
5. Keep it under 200 words
6. Include a clear call-to-action"""),
])


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


def _prompt_inputs(request: dict) -> dict:
    profile = request["profile"]
    return {
        "customer_id": profile.get("customer_id", "N/A"),
        "tenure": profile.get("tenure", 0),
        "contract": profile.get("contract", "N/A"),
        "monthly_charges": f"{profile.get('monthly_charges', 0):.2f}",
        "internet_service": profile.get("internet_service", "N/A"),
        "churn_probability": f"{request['churn_probability']:.1%}",
        "shap_drivers": json.dumps(request["shap_drivers"][:5], indent=2),
    }


def _template_message(request: dict) -> str:
    return generate_retention_message_template(
        request["profile"], request["churn_probability"], request["shap_drivers"]
    )


def _template_fallback(request: dict) -> str:
    logger.warning("LLM generation failed (%s), falling back to template.", request.get("error"))
    return _template_message(request)


def build_retention_chain(model: BaseChatModel) -> Runnable:
    """
    prompt -> model -> text, falling back to the template message if the model call fails.
    Input: {"profile": dict, "churn_probability": float, "shap_drivers": list}.
    """
    llm_chain = (
        RunnableLambda(_prompt_inputs) | RETENTION_PROMPT | model | StrOutputParser()
    ).with_config(run_name="retention_email")
    fallback = RunnableLambda(_template_fallback).with_config(run_name="retention_template")
    return llm_chain.with_fallbacks([fallback], exception_key="error")


def default_chat_model(api_key: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        max_tokens=500,
        max_retries=2,
        timeout=60,
    )


def _resolve_chain(api_key: Optional[str], model: Optional[BaseChatModel]) -> Optional[Runnable]:
    if model is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        model = default_chat_model(api_key)
    return build_retention_chain(model)


def generate_retention_message_llm(
    customer_profile: dict,
    churn_probability: float,
    shap_drivers: list,
    api_key: Optional[str] = None,
    model: Optional[BaseChatModel] = None,
) -> str:
    """
    Generate a personalized retention message with Claude through LangChain.
    Falls back to the template if no API key is available or the model call fails.
    """
    chain = _resolve_chain(api_key, model)
    if chain is None:
        return generate_retention_message_template(customer_profile, churn_probability, shap_drivers)
    return chain.invoke(
        {"profile": customer_profile, "churn_probability": churn_probability, "shap_drivers": shap_drivers},
        config={"run_name": "retention_message", "metadata": {"customer_id": customer_profile.get("customer_id")}},
    )


def batch_generate_messages(
    high_risk_customers: list,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    model: Optional[BaseChatModel] = None,
) -> list:
    """Generate retention messages for a batch of high-risk customers."""
    requests = [
        {"profile": c["profile"], "churn_probability": c["churn_prob"], "shap_drivers": c["shap_drivers"]}
        for c in high_risk_customers
    ]
    chain = _resolve_chain(api_key, model) if use_llm else None
    if chain is None:
        texts = [_template_message(r) for r in requests]
    else:
        texts = chain.batch(
            requests,
            config=[
                {"run_name": "retention_message", "max_concurrency": MAX_CONCURRENCY,
                 "metadata": {"customer_id": r["profile"].get("customer_id")}}
                for r in requests
            ],
        )
    return [
        {
            "customer_id": r["profile"].get("customer_id"),
            "churn_probability": r["churn_probability"],
            "message": text,
        }
        for r, text in zip(requests, texts)
    ]


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
