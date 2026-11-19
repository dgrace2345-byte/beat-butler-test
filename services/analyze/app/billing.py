"""Stripe Checkout and webhooks, with no SDK.

Three things are sold: a subscription, and two one-off coin packs that buy extra
uploads once the plan allowance runs out. Everything talks to Stripe over plain
HTTPS through `requests`, so the worker keeps its short dependency list and no
secret ever reaches the browser.

Configuration, all optional - a missing one disables that path with a 503 rather
than failing somewhere confusing:

    STRIPE_SECRET_KEY        sk_test_... / sk_live_...
    STRIPE_WEBHOOK_SECRET    whsec_...
    STRIPE_PRICE_STARTER     price_... for the $9/mo plan
    STRIPE_PRICE_PRO         price_... for the $19/mo plan
    STRIPE_PRICE_CREDITS_5   price_... for the 5-coin pack
    STRIPE_PRICE_CREDITS_20  price_... for the 20-coin pack
    SUPABASE_URL             project URL (the local stack's 54321 works)
    SUPABASE_ANON_KEY        public key, used to resolve the caller's token
    SUPABASE_SERVICE_KEY     service role key; writes plans and grants coins

Identity never comes from the request body: the caller's access token is resolved
against /auth/v1/user before a session exists, so nobody can buy for an account
they do not control.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

import requests

STRIPE_API = os.environ.get("STRIPE_API_BASE", "https://api.stripe.com/v1").rstrip("/")
TIMEOUT = 20
SIGNATURE_TOLERANCE = 300  # seconds, Stripe's own recommendation

PLAN_ENV = {"starter": "STRIPE_PRICE_STARTER", "pro": "STRIPE_PRICE_PRO"}
PACK_ENV = {"5": "STRIPE_PRICE_CREDITS_5", "20": "STRIPE_PRICE_CREDITS_20"}
PACK_CREDITS = {"5": 5, "20": 20}


class BillingNotConfigured(Exception):
    """Something is not set yet. The route answers 503 instead of guessing."""


class BillingFailed(Exception):
    """Stripe or Supabase refused, or the request could not be trusted."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def price_for(plan: str) -> str:
    env_name = PLAN_ENV.get(plan)
    if not env_name:
        raise BillingFailed(f"Unknown plan '{plan}'.")
    price = _env(env_name)
    if not price:
        raise BillingNotConfigured(f"{env_name} is not set, so {plan} cannot be bought yet.")
    return price


def price_for_pack(pack: str) -> str:
    env_name = PACK_ENV.get(pack)
    if not env_name:
        raise BillingFailed(f"Unknown coin pack '{pack}'.")
    price = _env(env_name)
    if not price:
        raise BillingNotConfigured(f"{env_name} is not set, so the {pack}-coin pack cannot be bought yet.")
    return price


def resolve_user(access_token: str) -> dict:
    """Who is this? Ask Supabase, never the request body."""
    base = _env("SUPABASE_URL").rstrip("/")
    anon = _env("SUPABASE_ANON_KEY")
    if not base or not anon:
        raise BillingNotConfigured("SUPABASE_URL / SUPABASE_ANON_KEY are not set.")
    try:
        res = requests.get(
            f"{base}/auth/v1/user",
            headers={"apikey": anon, "Authorization": f"Bearer {access_token}"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise BillingFailed(f"Could not reach Supabase auth: {exc}") from exc
    if res.status_code != 200:
        raise BillingFailed("That session is not valid. Sign in again.")
    user = res.json()
    if not user.get("id"):
        raise BillingFailed("Supabase returned no user for that token.")
    return user


def create_checkout_session(
    kind: str,
    plan: str,
    pack: str,
    user: dict,
    success_url: str,
    cancel_url: str,
) -> dict:
    """A hosted session for a subscription (kind="plan") or a coin pack (kind="credits")."""
    secret = _env("STRIPE_SECRET_KEY")
    if not secret:
        raise BillingNotConfigured("STRIPE_SECRET_KEY is not set, so checkout is off.")

    body = {
        "client_reference_id": user["id"],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "metadata[user_id]": user["id"],
    }
    if user.get("email"):
        body["customer_email"] = user["email"]

    if kind == "credits":
        credits = PACK_CREDITS.get(pack)
        if not credits:
            raise BillingFailed(f"Unknown coin pack '{pack}'.")
        body.update(
            {
                "mode": "payment",
                "line_items[0][price]": price_for_pack(pack),
                "line_items[0][quantity]": "1",
                "metadata[kind]": "credits",
                "metadata[pack]": pack,
                "metadata[credits]": str(credits),
            }
        )
    else:
        body.update(
            {
                "mode": "subscription",
                "line_items[0][price]": price_for(plan),
                "line_items[0][quantity]": "1",
                "metadata[kind]": "plan",
                "metadata[plan]": plan,
                "subscription_data[metadata][plan]": plan,
                "subscription_data[metadata][user_id]": user["id"],
            }
        )

    try:
        res = requests.post(
            f"{STRIPE_API}/checkout/sessions",
            data=body,
            headers={"Authorization": f"Bearer {secret}"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise BillingFailed(f"Could not reach Stripe: {exc}") from exc

    if res.status_code >= 300:
        raise BillingFailed(f"Stripe refused the session: HTTP {res.status_code} {res.text[:200]}")
    payload = res.json()
    return {"id": payload.get("id"), "url": payload.get("url")}


def verify_webhook(payload: bytes, header: str) -> dict:
    """Stripe's signature scheme: t=<timestamp>,v1=<hmac>[,v1=<hmac>...]."""
    secret = _env("STRIPE_WEBHOOK_SECRET")
    if not secret:
        raise BillingNotConfigured("STRIPE_WEBHOOK_SECRET is not set, so webhooks cannot be verified.")
    if not header:
        raise BillingFailed("Missing Stripe-Signature header.")

    timestamp = ""
    signatures: list[str] = []
    for item in header.split(","):
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key == "t":
            timestamp = value
        elif key == "v1":
            signatures.append(value)

    if not timestamp or not signatures:
        raise BillingFailed("Stripe-Signature is missing a timestamp or signature.")
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError as exc:
        raise BillingFailed("Stripe-Signature has an unreadable timestamp.") from exc
    if age > SIGNATURE_TOLERANCE:
        raise BillingFailed("Stripe-Signature is too old; check the server clock.")

    signed = f"{timestamp}.".encode() + payload
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, candidate) for candidate in signatures):
        raise BillingFailed("Stripe-Signature does not match this payload.")

    try:
        event = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BillingFailed("Webhook body is not JSON.") from exc
    if not isinstance(event, dict):
        raise BillingFailed("Webhook body is not an object.")
    return event


def handle_event(event: dict) -> list[dict]:
    """The changes this event describes; empty when it is not ours."""
    kind = str(event.get("type") or "")
    payload = event.get("data") if isinstance(event.get("data"), dict) else {}
    data = (payload or {}).get("object") or {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    user_id = data.get("client_reference_id") or metadata.get("user_id")

    if kind == "checkout.session.completed" and user_id:
        if data.get("mode") == "payment" or metadata.get("kind") == "credits":
            try:
                credits = int(metadata.get("credits") or PACK_CREDITS.get(str(metadata.get("pack")), 0))
            except (TypeError, ValueError):
                credits = 0
            return [{"user_id": user_id, "credits": credits}] if credits > 0 else []

        plan = str(metadata.get("plan") or "starter")
        return [{"user_id": user_id, "plan": plan}] if plan in PLAN_ENV else []

    if kind in {"customer.subscription.deleted", "customer.subscription.paused"} and user_id:
        return [{"user_id": user_id, "plan": "free"}]
    return []


def apply_action(action: dict) -> None:
    if "credits" in action:
        grant_credits(str(action["user_id"]), int(action["credits"]))
    else:
        apply_plan(str(action["user_id"]), str(action["plan"]))


def apply_plan(user_id: str, plan: str) -> None:
    """Write the plan with the service role key; RLS deliberately forbids this for users."""
    base = _env("SUPABASE_URL").rstrip("/")
    key = _env("SUPABASE_SERVICE_KEY")
    if not base or not key:
        raise BillingNotConfigured("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set.")
    if plan not in {"free", *PLAN_ENV}:
        raise BillingFailed(f"Unknown plan '{plan}'.")

    try:
        res = requests.patch(
            f"{base}/rest/v1/profiles?id=eq.{user_id}",
            json={"plan": plan},
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise BillingFailed(f"Could not reach Supabase: {exc}") from exc
    if res.status_code >= 300:
        raise BillingFailed(f"Could not update the profile: HTTP {res.status_code} {res.text[:200]}")


def grant_credits(user_id: str, credits: int) -> int:
    """Add coins through the service-role RPC; returns the new balance."""
    base = _env("SUPABASE_URL").rstrip("/")
    key = _env("SUPABASE_SERVICE_KEY")
    if not base or not key:
        raise BillingNotConfigured("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set.")

    try:
        res = requests.post(
            f"{base}/rest/v1/rpc/grant_beat_credits",
            json={"p_user": user_id, "p_amount": credits},
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise BillingFailed(f"Could not reach Supabase: {exc}") from exc
    if res.status_code >= 300:
        raise BillingFailed(f"Could not grant credits: HTTP {res.status_code} {res.text[:200]}")

    balance = res.json()
    return balance if isinstance(balance, int) else 0


def configured() -> dict[str, Any]:
    """What the upgrade path still needs, for the startup log and /health."""
    missing = [
        name
        for name in (
            "STRIPE_SECRET_KEY",
            "STRIPE_WEBHOOK_SECRET",
            "STRIPE_PRICE_STARTER",
            "STRIPE_PRICE_CREDITS_5",
            "SUPABASE_SERVICE_KEY",
        )
        if not _env(name)
    ]
    return {"ready": not missing, "missing": missing}
