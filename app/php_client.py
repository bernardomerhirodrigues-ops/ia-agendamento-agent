import json
import logging
import time
import httpx
from typing import Optional, Dict, Any, List

from .config import BASE_URL, API_KEY

logger = logging.getLogger(__name__)

# Máximo de caracteres por mensagem antes de quebrar (mais humano enviar em partes)
MAX_CHARS_PER_MESSAGE = 500
# Segundos entre mensagens quando enviamos várias (comportamento mais humano)
DELAY_BETWEEN_MESSAGES_SEC = 1.5


def _split_message_for_human_send(text: str) -> List[str]:
    """
    Divide o texto em partes por parágrafos (quebra de linha dupla) e, se necessário,
    por tamanho, para envio em múltiplas mensagens mais naturais.
    """
    if not text or not text.strip():
        return []
    normalized = text.replace("\r\n", "\n").strip()
    # Primeiro: separar por parágrafos (\n\n)
    parts = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    if not parts:
        return [normalized]
    result: List[str] = []
    for p in parts:
        if len(p) <= MAX_CHARS_PER_MESSAGE:
            result.append(p)
        else:
            # Parágrafo longo: quebrar por linha (\n)
            lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
            if not lines:
                lines = [p]
            current: List[str] = []
            current_len = 0
            for line in lines:
                if len(line) > MAX_CHARS_PER_MESSAGE:
                    if current:
                        result.append("\n".join(current))
                        current = []
                        current_len = 0
                    # Linha muito longa: quebrar por tamanho (tentando no espaço)
                    rest = line
                    while rest:
                        if len(rest) <= MAX_CHARS_PER_MESSAGE:
                            result.append(rest)
                            break
                        chunk = rest[:MAX_CHARS_PER_MESSAGE]
                        last_space = chunk.rfind(" ")
                        if last_space > MAX_CHARS_PER_MESSAGE // 2:
                            chunk = chunk[:last_space]
                            rest = rest[last_space:].lstrip()
                        else:
                            rest = rest[MAX_CHARS_PER_MESSAGE:]
                        result.append(chunk)
                elif current_len + len(line) + 1 <= MAX_CHARS_PER_MESSAGE and current:
                    current.append(line)
                    current_len += len(line) + 1
                else:
                    if current:
                        result.append("\n".join(current))
                    current = [line]
                    current_len = len(line)
            if current:
                result.append("\n".join(current))
    return result if result else [normalized]


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


def reserve_slot(date: str, time: str, candidate_name: str, responsible: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not BASE_URL or not API_KEY:
        return None
    payload: Dict[str, Any] = {"date": date, "time": time, "candidate_name": candidate_name}
    if responsible:
        payload["responsible"] = responsible
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{BASE_URL}/api/agent/reserve",
                headers=_headers(),
                json=payload,
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


def _send_one_message(phone: str, message: str) -> bool:
    """Envia uma única mensagem (uma chamada HTTP)."""
    if not BASE_URL or not API_KEY:
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


def send_whatsapp_message(phone: str, message: str) -> bool:
    """
    Envia a mensagem ao WhatsApp. Se for longa, divide por parágrafos (quebra de linha)
    e envia em várias mensagens com pequena pausa entre elas (comportamento mais humano).
    """
    chunks = _split_message_for_human_send(message)
    if not chunks:
        return True
    if len(chunks) == 1:
        return _send_one_message(phone, chunks[0])
    ok = True
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(DELAY_BETWEEN_MESSAGES_SEC)
        if not _send_one_message(phone, chunk):
            ok = False
    return ok
