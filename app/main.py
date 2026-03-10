import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

from .config import WEBHOOK_SECRET
from .db import (
    get_agent_config,
    get_conversation,
    try_mark_message_processed,
    add_to_message_buffer,
    get_buffered_messages,
    delete_buffer_for_phone,
    mark_message_processed,
    mark_human_takeover_chat,
    is_chat_human_takeover,
    get_phones_with_buffer_older_than_seconds,
)
from .agent import run_agent_with_openai
from .php_client import send_whatsapp_message

# Tarefas agendadas de flush do buffer por phone (canceladas quando chega nova mensagem)
_pending_flush_tasks: Dict[str, asyncio.Task] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="IA Agendamento Agent", version="1.0.0")


@app.on_event("startup")
async def startup_flush_orphaned_buffers():
    """Ao subir o serviço, processa buffers órfãos (ex.: após restart)."""
    try:
        config = get_agent_config()
        sec = max(0, int(config.get("message_buffer_seconds") or 0))
        if sec <= 0:
            return
        phones = get_phones_with_buffer_older_than_seconds(sec)
        for phone in phones:
            await _flush_buffer(phone)
        if phones:
            logger.info("startup: flush de %d buffer(s) órfão(s)", len(phones))
    except Exception as e:
        logger.warning("startup flush buffers: %s", e)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ia-agendamento-agent"}


async def _flush_buffer(phone: str) -> None:
    """Processa mensagens do buffer para o phone: junta textos, marca como processadas, chama agente e envia resposta."""
    global _pending_flush_tasks
    _pending_flush_tasks.pop(phone, None)
    try:
        conv = get_conversation(phone)
        if conv and (conv.get("flow_status") or "").strip() == "handed_to_human":
            delete_buffer_for_phone(phone)
            logger.info("webhook/whatsapp: flush cancelado (handed_to_human) phone=%s", phone[:6] + "****")
            return
        rows = get_buffered_messages(phone)
        if not rows:
            return
        aggregated = " ".join((r.get("text_content") or "").strip() for r in rows)
        first_name = (rows[-1].get("first_name") or "").strip() if rows else ""
        message_ids = [r.get("message_id") for r in rows if r.get("message_id")]
        for mid in message_ids:
            mark_message_processed(mid, phone)
        delete_buffer_for_phone(phone)
        if not aggregated.strip():
            reply = "Olá! Para agendar sua entrevista, por favor envie uma mensagem (por exemplo: 'Quero agendar entrevista')."
        else:
            try:
                reply = run_agent_with_openai(phone, first_name, aggregated.strip())
            except Exception as e:
                logger.exception("run_agent_with_openai failed in flush: %s", e)
                reply = "Desculpe, tive um problema técnico. Pode tentar novamente em instantes?"
        logger.info("webhook/whatsapp: flush buffer phone=%s messages=%d reply_len=%d", phone[:6] + "****", len(rows), len(reply))
        send_whatsapp_message(phone, reply)
    except Exception as e:
        logger.exception("flush_buffer failed for phone %s: %s", phone[:8], e)


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


def _normalize_phone(phone: str) -> str:
    """
    Normaliza número para formato E.164 Brasil (5585999999999).
    Remove prefixos inválidos e garante 55 para números brasileiros.
    """
    if not phone:
        return ""
    s = "".join(c for c in str(phone) if c.isdigit())
    # Remove prefixo 1 antes do 55 (ex.: 1558599999999 -> 5585999999999)
    if len(s) > 11 and s.startswith("155"):
        s = s[1:]
    # Remove zeros à esquerda antes do 55 (ex.: 0558599999999 -> 5585999999999)
    while len(s) > 11 and s.startswith("0"):
        s = s[1:]
    # Adiciona 55 se for número brasileiro (10+ dígitos) sem código do país
    if len(s) >= 10 and not s.startswith("55"):
        s = "55" + s
    # Corrige 5510XX (DDD 10 inválido): Treinee pode enviar 55+10+DDD+numero
    if len(s) >= 6 and s.startswith("5510"):
        s = "55" + s[4:]
    # Brasil: mobile 13 dígitos (55+2+9), fixo 12 (55+2+8)
    if len(s) > 13 and s.startswith("55"):
        s = s[:13]
    return s


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
                    elif "interactive" in msg:
                        inter = msg.get("interactive", {})
                        text = inter.get("list_reply", {}).get("title") or inter.get("button_reply", {}).get("title") or ""
                    profile = value.get("contacts", [{}])[0].get("profile", {}) if value.get("contacts") else {}
                    first_name = profile.get("first_name", "") or ""
                    phone = _normalize_phone(str(from_) if from_ is not None else "")
                    return {"message_id": mid, "phone": phone, "text": text, "first_name": first_name}
        return None

    def _to_text(val):
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return val.get("body") or val.get("text") or val.get("message") or ""
        return str(val)

    # Payload Treinee (formato key + message): sender_pn tem o número real, message.conversation o texto
    if "key" in body and "message" in body:
        key = body.get("key", {}) or {}
        msg = body.get("message", {}) or {}
        raw_phone = key.get("sender_pn") or key.get("remoteJid") or ""
        raw_phone = str(raw_phone).split("@")[0] if raw_phone else ""
        text = msg.get("conversation") or msg.get("extendedTextMessage", {}).get("text") or _to_text(msg)
        return {
            "message_id": key.get("id") or ("msg-" + uuid.uuid4().hex),
            "phone": _normalize_phone(raw_phone),
            "text": text if isinstance(text, str) else _to_text(text),
            "first_name": body.get("pushName", ""),
        }

    # Payload simples (teste ou outro provedor)
    if "message_id" in body and "from" in body:
        raw = body.get("text", body.get("body", body.get("message", "")))
        raw_phone = body.get("from", body.get("phone", ""))
        return {
            "message_id": body.get("message_id", body.get("id", "")),
            "phone": _normalize_phone(str(raw_phone) if raw_phone else ""),
            "text": _to_text(raw),
            "first_name": body.get("first_name", body.get("profile", {}).get("first_name", "") if isinstance(body.get("profile"), dict) else ""),
        }
    # Payload Treinee/provedor: phone ou from + texto em text/body/message/content
    if "phone" in body or "from" in body or "connectedPhone" in body:
        mid = body.get("messageId") or body.get("message_id") or body.get("id") or ("msg-" + uuid.uuid4().hex)
        # Treinee: fromMe=true → usar connectedPhone; fromMe=false → phone (pode ser 100649052156060@lid)
        if body.get("fromMe") is True and body.get("connectedPhone"):
            raw_phone = body["connectedPhone"]
        else:
            raw_phone = body.get("phone") or body.get("from") or body.get("sender") or ""
        if isinstance(body.get("contact"), dict):
            raw_phone = raw_phone or body["contact"].get("phone") or body["contact"].get("wa_id") or ""
        phone = _normalize_phone(str(raw_phone).split("@")[0] if raw_phone else "")
        # Treinee pode enviar texto em text, body, message, content ou interactive (botão/lista)
        raw_text = body.get("text") or body.get("body") or body.get("message") or body.get("content") or ""
        if isinstance(body.get("message"), dict):
            raw_text = raw_text or body["message"].get("body") or body["message"].get("text") or ""
        if not raw_text and isinstance(body.get("interactive"), dict):
            inter = body["interactive"]
            raw_text = inter.get("list_reply", {}).get("title") or inter.get("button_reply", {}).get("title") or ""
        text = _to_text(raw_text)
        return {
            "message_id": mid,
            "phone": phone,
            "text": text,
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

    # Ignorar mensagens enviadas PELO negócio (fromMe=true) e marcar chat como assumido por humano
    from_me = body.get("fromMe") if isinstance(body, dict) else None
    if from_me is None and isinstance(body.get("key"), dict):
        from_me = body["key"].get("fromMe")
    if from_me is True:
        chat_id = body.get("phone") or (body.get("key") or {}).get("remoteJid") or ""
        if chat_id:
            mark_human_takeover_chat(str(chat_id).strip())
            logger.info("webhook/whatsapp: fromMe=true, chat marcado como humano takeover chat_id=%s", str(chat_id)[:30])
        logger.info("webhook/whatsapp: ignorando mensagem fromMe=true (enviada pelo negócio)")
        return JSONResponse(content={"received": True}, status_code=200)

    # Só processar quando a mensagem foi enviada PARA o número configurado (evita responder em número pessoal)
    webhook_number = ""
    try:
        config = get_agent_config()
        webhook_number = (config and config.get("whatsapp_webhook_number")) or ""
        webhook_number = _normalize_phone(str(webhook_number).strip()) if webhook_number else ""
    except Exception:
        pass
    if webhook_number:
        connected = _normalize_phone(str(body.get("connectedPhone", "")).split("@")[0])
        if connected and connected != webhook_number:
            logger.info("webhook/whatsapp: ignorando mensagem enviada para outro número (connected=%s, esperado=%s)", connected[:8] + "****", webhook_number[:8] + "****")
            return JSONResponse(content={"received": True}, status_code=200)

    _text_preview = ""
    if isinstance(body, dict):
        for k in ("text", "body", "message", "content"):
            v = body.get(k)
            if v is not None:
                _text_preview = str(v)[:80] if not isinstance(v, dict) else str(v.get("body") or v.get("text") or "")[:80]
                break
    logger.info("webhook/whatsapp received", extra={"has_body": bool(body), "body_keys": list(body.keys()) if isinstance(body, dict) else [], "text_source_preview": _text_preview})

    payload = _extract_whatsapp_payload(body)
    if not payload:
        return JSONResponse(content={"received": True}, status_code=200)

    message_id = payload.get("message_id") or ""
    phone = payload.get("phone") or ""
    chat_id = body.get("phone") or (body.get("key") or {}).get("remoteJid") or ""

    # Não processar se este chat foi assumido por humano (mensagem manual do negócio)
    if chat_id and is_chat_human_takeover(str(chat_id).strip()):
        logger.info("webhook/whatsapp: chat já em atendimento humano, ignorando chat_id=%s", str(chat_id)[:30])
        return JSONResponse(content={"received": True}, status_code=200)
    # Não processar se o candidato já confirmou que quer falar com atendente
    conv = get_conversation(phone)
    if conv and (conv.get("flow_status") or "").strip() == "handed_to_human":
        logger.info("webhook/whatsapp: conversa handed_to_human, ignorando phone=%s", phone[:6] + "****" if len(phone) > 6 else "****")
        return JSONResponse(content={"received": True}, status_code=200)
    raw_text = payload.get("text")
    if isinstance(raw_text, dict):
        text = (raw_text.get("body") or raw_text.get("text") or "").strip()
    else:
        text = (str(raw_text or "")).strip()
    first_name = (payload.get("first_name") or "").strip()

    _mid = (message_id[:25] + "...") if len(message_id) > 25 else (message_id or "(vazio)")
    _plen = len(phone)
    _pmask = (phone[:6] + "****") if _plen > 6 else "****"
    raw_from_body = body.get("phone") or body.get("from") or body.get("sender") or ""
    logger.info("webhook/whatsapp: payload extraído message_id=%s text_len=%d phone_raw=%s phone_normalized=%s", _mid, len(text), str(raw_from_body)[:20], _pmask)

    if not phone:
        return JSONResponse(content={"received": True}, status_code=200)

    buffer_seconds = 0
    try:
        config = get_agent_config()
        buffer_seconds = max(0, int(config.get("message_buffer_seconds") or 0))
    except Exception:
        pass

    if buffer_seconds > 0:
        # Buffer ativo: adiciona ao buffer e agenda flush (ou reinicia o timer)
        added = add_to_message_buffer(phone, message_id or ("buf-" + uuid.uuid4().hex), text, first_name)
        if not added:
            logger.info("webhook/whatsapp: mensagem já no buffer (duplicata), message_id=%s", (message_id or "")[:30])
            return JSONResponse(content={"received": True}, status_code=200)
        global _pending_flush_tasks
        if phone in _pending_flush_tasks:
            _pending_flush_tasks[phone].cancel()
        _pending_flush_tasks[phone] = asyncio.create_task(_schedule_flush(phone, buffer_seconds))
        logger.info("webhook/whatsapp: mensagem adicionada ao buffer, flush em %ds", buffer_seconds)
        return JSONResponse(content={"received": True}, status_code=200)

    # Sem buffer: idempotência e processamento imediato
    if message_id:
        if not try_mark_message_processed(message_id, phone):
            logger.info("webhook/whatsapp: duplicate message_id ignored", extra={"message_id": message_id[:30] if message_id else "(vazio)"})
            return JSONResponse(content={"received": True}, status_code=200)

    if not text:
        reply = "Olá! Para agendar sua entrevista, por favor envie uma mensagem (por exemplo: 'Quero agendar entrevista')."
        logger.info("webhook/whatsapp: enviando resposta padrão (texto vazio)", extra={"phone_masked": phone[:6] + "****" if len(phone) > 6 else "****"})
        send_whatsapp_message(phone, reply)
        return JSONResponse(content={"received": True}, status_code=200)

    try:
        reply = run_agent_with_openai(phone, first_name, text)
    except Exception as e:
        logger.exception("run_agent_with_openai failed: %s", e)
        reply = "Desculpe, tive um problema técnico. Pode tentar novamente em instantes?"
    logger.info("webhook/whatsapp: enviando resposta via PHP", extra={"phone_masked": phone[:6] + "****" if len(phone) > 6 else "****", "reply_len": len(reply)})
    send_whatsapp_message(phone, reply)
    return JSONResponse(content={"received": True}, status_code=200)


async def _schedule_flush(phone: str, seconds: int) -> None:
    """Aguarda seconds e chama _flush_buffer(phone)."""
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        pass
    await _flush_buffer(phone)
