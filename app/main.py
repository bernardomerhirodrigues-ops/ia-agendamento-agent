import json
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

from .config import WEBHOOK_SECRET
from .db import is_message_processed, mark_message_processed
from .agent import run_agent
from .php_client import send_whatsapp_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="IA Agendamento Agent", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ia-agendamento-agent"}


@app.post("/webhook/test")
async def webhook_test(
    request: Request,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    secret_required = (WEBHOOK_SECRET or "").strip()
    if secret_required and (x_webhook_secret or "").strip() != secret_required:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    body = await request.json() if await request.body() else {}
    logger.info("webhook_test received", extra={"body": body})
    return {"received": True, "message": "Teste recebido com sucesso."}


def _extract_whatsapp_payload(body: dict) -> Optional[dict]:
    """
    Extrai message_id, phone (from), text, first_name de um payload genérico.
    Suporta formato Meta Cloud API e estruturas simples.
    """
    # Meta Cloud API: entry -> changes -> value -> messages
    if "entry" in body:
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    mid = msg.get("id")
                    from_ = msg.get("from")
                    text = ""
                    if "text" in msg:
                        text = msg.get("text", {}).get("body", "")
                    elif "button" in msg:
                        text = msg.get("button", {}).get("text", "")
                    profile = value.get("contacts", [{}])[0].get("profile", {}) if value.get("contacts") else {}
                    first_name = profile.get("first_name", "") or ""
                    return {"message_id": mid, "phone": from_, "text": text, "first_name": first_name}
        return None

    def _to_text(val):
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return val.get("body") or val.get("text") or ""
        return str(val)

    # Payload simples (teste ou outro provedor)
    if "message_id" in body and "from" in body:
        raw = body.get("text", body.get("body", body.get("message", "")))
        return {
            "message_id": body.get("message_id", body.get("id", "")),
            "phone": body.get("from", body.get("phone", "")),
            "text": _to_text(raw),
            "first_name": body.get("first_name", body.get("profile", {}).get("first_name", "") if isinstance(body.get("profile"), dict) else ""),
        }
    if "phone" in body and "text" in body:
        return {
            "message_id": body.get("message_id", "test-" + str(hash(body.get("phone", "")))),
            "phone": body["phone"],
            "text": _to_text(body["text"]),
            "first_name": body.get("first_name", ""),
        }
    return None


@app.post("/webhook/whatsapp")
async def webhook_whatsapp(
    request: Request,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    secret_required = (WEBHOOK_SECRET or "").strip()
    if secret_required and (x_webhook_secret or "").strip() != secret_required:
        logger.warning("webhook/whatsapp: invalid or missing secret")
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}

    logger.info("webhook/whatsapp received", extra={"has_body": bool(body)})

    # Verificação de assinatura Meta (hub.verify_token no GET não usado aqui; POST é o evento)
    # Alguns provedores enviam GET para verificação da URL; ignoramos se for GET
    payload = _extract_whatsapp_payload(body)
    if not payload:
        return JSONResponse(content={"received": True}, status_code=200)

    message_id = payload.get("message_id") or ""
    phone = payload.get("phone") or ""
    raw_text = payload.get("text")
    if isinstance(raw_text, dict):
        text = (raw_text.get("body") or raw_text.get("text") or "").strip()
    else:
        text = (str(raw_text or "")).strip()
    first_name = (payload.get("first_name") or "").strip()

    _mid = (message_id[:25] + "...") if len(message_id) > 25 else (message_id or "(vazio)")
    logger.info("webhook/whatsapp: payload extraído message_id=%s text_len=%d", _mid, len(text))

    if not phone:
        return JSONResponse(content={"received": True}, status_code=200)

    # Idempotência
    if message_id and is_message_processed(message_id):
        logger.info("webhook/whatsapp: duplicate message_id ignored", extra={"message_id": message_id[:30] if message_id else "(vazio)"})
        return JSONResponse(content={"received": True}, status_code=200)

    if message_id:
        mark_message_processed(message_id, phone)

    if not text:
        reply = "Olá! Para agendar sua entrevista, por favor envie uma mensagem (por exemplo: 'Quero agendar entrevista')."
        logger.info("webhook/whatsapp: enviando resposta padrão (texto vazio)", extra={"phone_masked": phone[:6] + "****" if len(phone) > 6 else "****"})
        send_whatsapp_message(phone, reply)
        return JSONResponse(content={"received": True}, status_code=200)

    try:
        reply = run_agent(phone, first_name, text)
    except Exception as e:
        logger.exception("run_agent failed: %s", e)
        reply = "Desculpe, tive um problema técnico. Pode tentar novamente em instantes?"
    logger.info("webhook/whatsapp: enviando resposta via PHP", extra={"phone_masked": phone[:6] + "****" if len(phone) > 6 else "****", "reply_len": len(reply)})
    send_whatsapp_message(phone, reply)
    return JSONResponse(content={"received": True}, status_code=200)
