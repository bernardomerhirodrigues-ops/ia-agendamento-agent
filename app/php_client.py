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
        return False
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{BASE_URL}/api/agent/send-message",
                headers=_headers(),
                json={"phone": phone, "message": message},
            )
            r.raise_for_status()
            data = r.json()
            return data.get("success", False)
    except Exception as e:
        logger.exception("send_whatsapp_message failed: %s", e)
        return False
