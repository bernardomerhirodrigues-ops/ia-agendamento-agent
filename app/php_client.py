import json
import logging
import httpx
from typing import Optional, Dict, Any

from .config import BASE_URL, API_KEY

logger = logging.getLogger(__name__)


def _headers() -> dict:
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def get_next_slot() -> Optional[Dict[str, Any]]:
    if not BASE_URL or not API_KEY:
        logger.warning("BASE_URL or API_KEY not set")
        return None
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(f"{BASE_URL}/api/agent/next-slot", headers=_headers())
            r.raise_for_status()
            data = r.json()
            if data.get("success") and data.get("slot"):
                return data["slot"]
            return None
    except Exception as e:
        logger.exception("get_next_slot failed: %s", e)
        return None


def reserve_slot(date: str, time: str, candidate_name: str) -> Optional[Dict[str, Any]]:
    if not BASE_URL or not API_KEY:
        return None
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{BASE_URL}/api/agent/reserve",
                headers=_headers(),
                json={"date": date, "time": time, "candidate_name": candidate_name},
            )
            r.raise_for_status()
            data = r.json()
            if data.get("success"):
                return data
            return None
    except Exception as e:
        logger.exception("reserve_slot failed: %s", e)
        return None


def get_entrevistador() -> Optional[str]:
    if not BASE_URL or not API_KEY:
        return None
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{BASE_URL}/api/agent/entrevistador", headers=_headers())
            r.raise_for_status()
            data = r.json()
            if data.get("success"):
                return data.get("nome_entrevistador") or ""
            return None
    except Exception as e:
        logger.exception("get_entrevistador failed: %s", e)
        return None


def send_whatsapp_message(phone: str, message: str) -> bool:
    if not BASE_URL or not API_KEY:
        logger.warning("send_whatsapp_message: BASE_URL or API_KEY not set")
        return False
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{BASE_URL}/api/agent/send-message",
                headers=_headers(),
                json={"phone": phone, "message": message},
            )
            raw = r.text
            if r.status_code >= 400:
                try:
                    data = json.loads(raw) if raw else {}
                    err_msg = data.get("message", raw[:200] if raw else f"HTTP {r.status_code}")
                except json.JSONDecodeError:
                    err_msg = raw[:200] if raw else f"HTTP {r.status_code}"
                logger.warning("send_whatsapp_message: PHP retornou HTTP %d - %s", r.status_code, err_msg)
                return False
            if not raw or not raw.strip():
                logger.warning("send_whatsapp_message: PHP retornou corpo vazio (HTTP %d)", r.status_code)
                return False
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "send_whatsapp_message: PHP retornou corpo inválido (não-JSON): %s...",
                    raw[:100] if len(raw) > 100 else raw,
                )
                return False
            ok = data.get("success", False)
            if not ok:
                logger.warning(
                    "send_whatsapp_message: PHP returned success=false - %s",
                    data.get("message", "(sem mensagem)"),
                )
            return ok
    except httpx.HTTPStatusError as e:
        logger.warning("send_whatsapp_message: HTTP error %d - %s", e.response.status_code, e.response.text[:200] if e.response.text else "")
        return False
    except Exception as e:
        logger.exception("send_whatsapp_message failed: %s", e)
        return False
