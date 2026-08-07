"""Traction — autonomous account retention + upsell agent for B2B SaaS.

Watches every account in a portfolio, flags churn risk and upsell readiness
with deterministic rules, and (in later stages) drafts outreach grounded in
that specific customer's history.

Honesty rule for the whole project: every account in here is seeded and
simulated. Nothing is observed from a real customer, and any payload that
carries a money figure is tagged ``"simulated": true``.
"""

__version__ = "0.1.0"

__all__ = ["config", "seed", "scoring", "store"]
