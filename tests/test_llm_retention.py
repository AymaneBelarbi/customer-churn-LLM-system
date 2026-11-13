import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_retention import (  # noqa: E402
    RETENTION_PROMPT,
    _prompt_inputs,
    batch_generate_messages,
    generate_retention_message_llm,
    generate_retention_message_template,
)

PROFILE = {
    "customer_id": "7590-VHVEG",
    "tenure": 3,
    "contract": "Month-to-month",
    "monthly_charges": 89.5,
    "internet_service": "Fiber optic",
}
DRIVERS = [
    {"feature": "Contract_Month-to-month", "shap_impact": 0.45, "direction": "increases"},
    {"feature": "tenure", "shap_impact": -0.32, "direction": "increases"},
]


class FailingChatModel(GenericFakeChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("upstream 529")


def test_prompt_renders_profile_and_shap_drivers():
    text = RETENTION_PROMPT.format(
        **_prompt_inputs({"profile": PROFILE, "churn_probability": 0.854, "shap_drivers": DRIVERS})
    )
    assert "Customer ID: 7590-VHVEG" in text
    assert "Monthly Charges: $89.50" in text
    assert "Churn Probability: 85.4%" in text
    assert '"feature": "Contract_Month-to-month"' in text


def test_llm_message_comes_from_the_model():
    model = FakeListChatModel(responses=["Hi 7590-VHVEG, 20% off annual plans."])
    message = generate_retention_message_llm(PROFILE, 0.85, DRIVERS, model=model)
    assert message == "Hi 7590-VHVEG, 20% off annual plans."


def test_model_failure_falls_back_to_template(caplog):
    model = FailingChatModel(messages=iter([]))
    message = generate_retention_message_llm(PROFILE, 0.85, DRIVERS, model=model)
    assert message == generate_retention_message_template(PROFILE, 0.85, DRIVERS)
    assert "upstream 529" in caplog.text


def test_no_api_key_uses_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert generate_retention_message_llm(PROFILE, 0.7, DRIVERS) == generate_retention_message_template(
        PROFILE, 0.7, DRIVERS
    )


@pytest.mark.parametrize("use_llm", [True, False])
def test_batch_keeps_customer_order(use_llm):
    customers = [
        {"profile": {**PROFILE, "customer_id": f"C-{i}"}, "churn_prob": 0.6 + i / 10, "shap_drivers": DRIVERS}
        for i in range(3)
    ]
    model = FakeListChatModel(responses=["email"]) if use_llm else None
    results = batch_generate_messages(customers, use_llm=use_llm, model=model)
    assert [r["customer_id"] for r in results] == ["C-0", "C-1", "C-2"]
    assert all(r["message"] for r in results)
    if use_llm:
        assert {r["message"] for r in results} == {"email"}
