import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

from .config import WEBHOOK_SECRET
from .db import (
    get_agent_config,
    get_conversation,
    get_memory,
    try_mark_message_processed,
    add_to_message_buffer,
    get_buffered_messages,
    delete_buffer_for_phone,
    mark_message_processed,
    mark_human_takeover_chat,
    is_chat_human_takeover,
    get_phones_with_buffer_older_than_seconds,
    reset_handed_to_human,
    update_candidate_info,
)
from .candidate_info import extract_candidate_info
from .agent import run_agent_with_openai
from .config import BASE_URL
from .media_processor import (
    download_file,
    transcribe_audio,
    describe_image,
    extract_document_text,
)
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
                info = extract_candidate_info(aggregated)
                if info.get("candidate_age") is not None or info.get("study_shift") or info.get("study_hours"):
                    update_candidate_info(phone, candidate_age=info.get("candidate_age"), study_shift=info.get("study_shift"), study_hours=info.get("study_hours"))
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


# Fuso usado para dias/horários ativos (Painel de Controle). Render roda em UTC; o usuário configura em horário de Brasília.
_ACTIVE_SCHEDULE_TZ = ZoneInfo("America/Sao_Paulo")


def _is_within_active_schedule(config: Optional[Dict[str, Any]]) -> bool:
    """
    Verifica se o momento atual está dentro dos dias e horários ativos configurados.
    Usa horário de Brasília (America/Sao_Paulo); o servidor Render roda em UTC.
    active_days: 1=Segunda a 7=Domingo (PHP date('N')), separados por vírgula.
    """
    if not config:
        return True
    active_days_raw = (config.get("active_days") or "").strip()
    start_raw = (config.get("active_time_start") or "").strip()
    end_raw = (config.get("active_time_end") or "").strip()
    if not active_days_raw and not start_raw and not end_raw:
        return True
    now = datetime.now(_ACTIVE_SCHEDULE_TZ)
    # Dia da semana: isoweekday() 1=Segunda .. 7=Domingo (igual ao PHP date('N'))
    if active_days_raw:
        allowed = [int(x) for x in active_days_raw.split(",") if x.strip().isdigit()]
        if allowed and now.isoweekday() not in allowed:
            return False
    if start_raw or end_raw:
        try:
            start_t = dt_time(0, 0)
            end_t = dt_time(23, 59, 59)
            if start_raw:
                parts = start_raw.split(":")
                start_t = dt_time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
            if end_raw:
                parts = end_raw.split(":")
                end_t = dt_time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
            now_t = now.time()
            if start_t <= end_t:
                if not (start_t <= now_t <= end_t):
                    return False
            else:
                # intervalo atravessa meia-noite (ex.: 22:00 a 06:00)
                if not (now_t >= start_t or now_t <= end_t):
                    return False
        except (ValueError, IndexError):
            pass
    return True


def _extract_connected_phone(body: dict) -> str:
    """
    Extrai o número da instância/canal (uso legado; em alguns provedores connectedPhone = remetente).
    Ver _extract_receiver_phone para o número que RECEBEU a mensagem.
    """
    if not body or not isinstance(body, dict):
        return ""
    candidates = [
        body.get("connectedPhone"),
        body.get("instance"),
        body.get("to"),
        body.get("receiver"),
        body.get("receiverPhone"),
        body.get("destination"),
    ]
    data = body.get("data")
    if isinstance(data, dict):
        candidates.extend([
            data.get("connectedPhone"),
            data.get("instance"),
            data.get("to"),
            data.get("receiver"),
        ])
    for raw in candidates:
        if not raw:
            continue
        s = str(raw).strip().split("@")[0]
        normalized = _normalize_phone(s)
        if len(normalized) >= 12 and normalized.startswith("55"):
            return normalized
    return ""


def _extract_receiver_phone(body: dict) -> str:
    """
    Extrai o número que RECEBEU a mensagem (instância/canal de destino).
    Não usa connectedPhone, pois em vários provedores connectedPhone = quem ENVIOU (remetente).
    Usa apenas: receiver, to, receiverPhone, instance (se for número), destination, me.
    """
    if not body or not isinstance(body, dict):
        return ""
    candidates = [
        body.get("receiver"),
        body.get("to"),
        body.get("receiverPhone"),
        body.get("destination"),
        body.get("me"),
        body.get("instance"),  # em alguns provedores instance = número da instância que recebeu
    ]
    data = body.get("data")
    if isinstance(data, dict):
        candidates.extend([
            data.get("receiver"),
            data.get("to"),
            data.get("receiverPhone"),
            data.get("instance"),
        ])
    for raw in candidates:
        if not raw:
            continue
        s = str(raw).strip().split("@")[0]
        normalized = _normalize_phone(s)
        if len(normalized) >= 12 and normalized.startswith("55"):
            return normalized
    return ""


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
                    msg_type = msg.get("type")
                    text = ""
                    # Texto comum, botões e interativos
                    if "text" in msg:
                        text = msg.get("text", {}).get("body", "")
                    elif "button" in msg:
                        text = msg.get("button", {}).get("text", "")
                    elif "interactive" in msg:
                        inter = msg.get("interactive", {})
                        text = inter.get("list_reply", {}).get("title") or inter.get("button_reply", {}).get("title") or ""
                    # Mídia: mapeia para marcadores e extrai URL/ID para o agente ler o conteúdo
                    media_url = None
                    media_id = None
                    media_mime = None
                    media_filename = None
                    if msg_type == "audio":
                        text = "[AUDIO]"
                        obj = msg.get("audio") or {}
                        media_id = obj.get("id")
                        media_url = obj.get("url") or obj.get("directLink") or obj.get("link")
                    elif msg_type == "image":
                        text = "[IMAGE]"
                        obj = msg.get("image") or {}
                        media_id = obj.get("id")
                        media_url = obj.get("url") or obj.get("directLink") or obj.get("link")
                    elif msg_type == "document":
                        doc = msg.get("document", {}) or {}
                        mime = (doc.get("mime_type") or "").lower()
                        media_id = doc.get("id")
                        media_url = doc.get("url") or doc.get("directLink") or doc.get("link")
                        media_mime = doc.get("mime_type")
                        media_filename = doc.get("filename")
                        if "pdf" in mime:
                            text = "[PDF]"
                        elif "excel" in mime or "spreadsheet" in mime or mime in ("application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
                            text = "[EXCEL]"
                        elif "word" in mime or mime in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                            text = "[WORD]"
                        else:
                            text = "[DOCUMENT]"
                    profile = value.get("contacts", [{}])[0].get("profile", {}) if value.get("contacts") else {}
                    first_name = profile.get("first_name", "") or ""
                    phone = _normalize_phone(str(from_) if from_ is not None else "")
                    out = {"message_id": mid, "phone": phone, "text": text, "first_name": first_name}
                    if media_url or media_id:
                        out["media_url"], out["media_id"] = media_url, media_id
                        if media_mime:
                            out["media_mime"] = media_mime
                        if media_filename:
                            out["media_filename"] = media_filename
                    return out
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
        # Texto comum
        text = msg.get("conversation") or msg.get("extendedTextMessage", {}).get("text") or ""
        # Mídia (áudio, imagem, documento) – converte para marcadores e extrai URL/ID
        media_url, media_id, media_mime, media_filename = None, None, None, None
        if not text and isinstance(msg, dict):
            if "audioMessage" in msg:
                text = "[AUDIO]"
                obj = msg.get("audioMessage") or {}
                media_id = obj.get("id")
                media_url = obj.get("url") or obj.get("directLink") or obj.get("link") or obj.get("directPath")
            elif "imageMessage" in msg:
                text = "[IMAGE]"
                obj = msg.get("imageMessage") or {}
                media_id = obj.get("id")
                media_url = obj.get("url") or obj.get("directLink") or obj.get("link") or obj.get("directPath")
            elif "documentMessage" in msg:
                doc = msg.get("documentMessage") or {}
                mime = (doc.get("mimetype") or "").lower()
                media_id = doc.get("id")
                media_url = doc.get("url") or doc.get("directLink") or doc.get("link") or doc.get("directPath")
                media_mime = doc.get("mimetype")
                media_filename = doc.get("fileName") or doc.get("filename")
                if "pdf" in mime:
                    text = "[PDF]"
                elif "excel" in mime or "spreadsheet" in mime or mime in ("application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
                    text = "[EXCEL]"
                elif "word" in mime or mime in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                    text = "[WORD]"
                else:
                    text = "[DOCUMENT]"
        if not text:
            text = _to_text(msg)
        out = {
            "message_id": key.get("id") or ("msg-" + uuid.uuid4().hex),
            "phone": _normalize_phone(raw_phone),
            "text": text if isinstance(text, str) else _to_text(text),
            "first_name": body.get("pushName", ""),
        }
        if media_url or media_id:
            out["media_url"], out["media_id"] = media_url, media_id
            if media_mime:
                out["media_mime"] = media_mime
            if media_filename:
                out["media_filename"] = media_filename
        return out

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
        media_url, media_id, media_mime, media_filename = None, None, None, None
        if isinstance(body.get("message"), dict):
            msg_obj = body["message"]
            raw_text = raw_text or msg_obj.get("body") or msg_obj.get("text") or ""
            if not raw_text:
                if "audioMessage" in msg_obj or body.get("type") == "audio":
                    raw_text = "[AUDIO]"
                    obj = msg_obj.get("audioMessage") or {}
                    media_id = obj.get("id") or body.get("mediaId")
                    media_url = obj.get("url") or obj.get("directLink") or obj.get("link") or body.get("audioUrl") or body.get("mediaUrl")
                elif "imageMessage" in msg_obj or body.get("type") == "image":
                    raw_text = "[IMAGE]"
                    obj = msg_obj.get("imageMessage") or {}
                    media_id = obj.get("id") or body.get("mediaId")
                    media_url = obj.get("url") or obj.get("directLink") or obj.get("link") or body.get("imageUrl") or body.get("mediaUrl")
                elif "documentMessage" in msg_obj or body.get("type") == "document":
                    doc = msg_obj.get("documentMessage") or {}
                    mime = (doc.get("mimetype") or body.get("mimeType") or "").lower()
                    media_id = doc.get("id") or body.get("mediaId")
                    media_url = doc.get("url") or doc.get("directLink") or doc.get("link") or body.get("documentUrl") or body.get("mediaUrl")
                    media_mime = doc.get("mimetype") or body.get("mimeType")
                    media_filename = doc.get("fileName") or doc.get("filename") or body.get("filename")
                    if "pdf" in mime:
                        raw_text = "[PDF]"
                    elif "excel" in mime or "spreadsheet" in mime or mime in ("application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"):
                        raw_text = "[EXCEL]"
                    elif "word" in mime or mime in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                        raw_text = "[WORD]"
                    else:
                        raw_text = "[DOCUMENT]"
        if not raw_text and isinstance(body.get("interactive"), dict):
            inter = body["interactive"]
            raw_text = inter.get("list_reply", {}).get("title") or inter.get("button_reply", {}).get("title") or ""
        text = _to_text(raw_text)
        out = {
            "message_id": mid,
            "phone": phone,
            "text": text,
            "first_name": body.get("first_name", ""),
        }
        if media_url or media_id:
            out["media_url"], out["media_id"] = media_url, media_id
            if media_mime:
                out["media_mime"] = media_mime
            if media_filename:
                out["media_filename"] = media_filename
        return out
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

    # Log completo do webhook recebido (útil para depuração de formato do provedor/Treinee)
    try:
        raw_json = json.dumps(body, ensure_ascii=False, default=str)
        # Limite para não estourar log (ex.: mídia com URLs grandes); ajuste se precisar ver tudo
        max_log_len = 8000
        if len(raw_json) > max_log_len:
            raw_json = raw_json[:max_log_len] + f"... [truncado, total {len(raw_json)} chars]"
        logger.info("webhook/whatsapp raw body: %s", raw_json)
    except Exception as e:
        logger.warning("webhook/whatsapp: falha ao serializar body para log: %s", e)

    # Provedor pode usar fromMe em sentidos opostos: (A) fromMe=true = enviada pelo negócio → ignorar;
    # (B) fromMe=true = recebida do contato → processar. Aqui tratamos (B): ignorar só quando fromMe=false (enviada pelo negócio).
    from_me = body.get("fromMe") if isinstance(body, dict) else None
    if from_me is None and isinstance(body.get("key"), dict):
        from_me = body["key"].get("fromMe")
    # fromMe=false = mensagem enviada PELO negócio (outgoing) → ignorar e eventualmente marcar human takeover
    if from_me is False:
        chat_id = body.get("phone") or (body.get("key") or {}).get("remoteJid") or ""
        if chat_id:
            chat_id = str(chat_id).strip()
            candidate_phone = _normalize_phone((chat_id.split("@")[0] or "").strip())
            has_prior_chat = (
                len(candidate_phone) >= 12
                and candidate_phone.startswith("55")
                and get_memory(candidate_phone, limit=1)
            )
            if has_prior_chat:
                mark_human_takeover_chat(chat_id)
                logger.info("webhook/whatsapp: fromMe=false e conversa já existia, chat marcado como humano takeover")
        logger.info("webhook/whatsapp: ignorando mensagem fromMe=false (enviada pelo negócio)")
        return JSONResponse(content={"received": True}, status_code=200)
    # fromMe=true ou None: tratar como mensagem do contato (processar e responder)

    # Obter config (agente desativado = sem config)
    try:
        config = get_agent_config()
    except Exception:
        config = None
    if not config:
        logger.info("webhook/whatsapp: agente desativado ou sem config, ignorando")
        return JSONResponse(content={"received": True}, status_code=200)

    # Só processar quando a mensagem foi RECEBIDA pelo número configurado (evita responder em número pessoal).
    # Atenção: em alguns provedores "connectedPhone" é quem ENVIOU (remetente), não quem recebeu. Por isso
    # usamos apenas campos que indiquem o receptor/instância (receiver, to, instance, etc.), não connectedPhone.
    webhook_number = _normalize_phone(str((config.get("whatsapp_webhook_number") or "")).strip())
    if webhook_number:
        receiver = _extract_receiver_phone(body)
        if receiver and receiver != webhook_number:
            logger.info("webhook/whatsapp: ignorando mensagem recebida em outro número (receiver=%s, esperado=%s)", receiver[:8] + "****", webhook_number[:8] + "****")
            return JSONResponse(content={"received": True}, status_code=200)

    # Respeitar dias e horários ativos (Painel de Controle)
    if not _is_within_active_schedule(config):
        logger.info("webhook/whatsapp: fora do horário/dias ativos configurados, ignorando")
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
    # Não processar se o candidato já confirmou que quer falar com atendente (respeitando bloqueio temporário)
    conv = get_conversation(phone)
    if conv and (conv.get("flow_status") or "").strip() == "handed_to_human":
        block_minutes = 0
        try:
            config = get_agent_config()
            block_minutes = max(0, int(config.get("human_takeover_block_minutes") or 0))
        except Exception:
            pass
        if block_minutes <= 0:
            logger.info("webhook/whatsapp: conversa handed_to_human (bloqueio permanente), ignorando phone=%s", phone[:6] + "****" if len(phone) > 6 else "****")
            return JSONResponse(content={"received": True}, status_code=200)
        handed_at = conv.get("handed_at")
        if handed_at:
            if isinstance(handed_at, str):
                try:
                    handed_at = datetime.fromisoformat(handed_at.replace("Z", "+00:00"))
                except ValueError:
                    handed_at = None
            if handed_at and handed_at.tzinfo is None:
                handed_at = handed_at.replace(tzinfo=timezone.utc)
            if handed_at:
                expiry = handed_at + timedelta(minutes=block_minutes)
                now = datetime.now(timezone.utc)
                if now >= expiry:
                    reset_handed_to_human(phone)
                    logger.info("webhook/whatsapp: bloqueio handed_to_human expirado, reativando agente phone=%s", phone[:6] + "****" if len(phone) > 6 else "****")
                else:
                    logger.info("webhook/whatsapp: conversa handed_to_human (bloqueio %s min), ignorando phone=%s", block_minutes, phone[:6] + "****" if len(phone) > 6 else "****")
                    return JSONResponse(content={"received": True}, status_code=200)
        else:
            logger.info("webhook/whatsapp: conversa handed_to_human (sem handed_at), ignorando phone=%s", phone[:6] + "****" if len(phone) > 6 else "****")
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

    # Mídia: tentar baixar e "ler" (transcrição, descrição ou extração de texto) para passar ao agente
    media_url = payload.get("media_url")
    media_id = payload.get("media_id")
    if not media_url and media_id and BASE_URL:
        media_url = BASE_URL.rstrip("/") + "/api/agent/media?id=" + str(media_id).strip()
    if text in ("[AUDIO]", "[IMAGE]", "[PDF]", "[EXCEL]", "[WORD]", "[DOCUMENT]") and media_url:
        data, content_type = download_file(media_url)
        if data:
            try:
                agent_config = get_agent_config()
                api_key = (agent_config and agent_config.get("openai_api_key")) or os.getenv("OPENAI_API_KEY", "")
                if text == "[AUDIO]":
                    extracted = transcribe_audio(data, api_key, content_type)
                    text = f"O usuário enviou um áudio. Transcrição: {extracted}"
                elif text == "[IMAGE]":
                    extracted = describe_image(data, api_key, content_type)
                    text = f"O usuário enviou uma imagem. Descrição: {extracted}"
                else:
                    mime = payload.get("media_mime") or content_type or ""
                    filename = payload.get("media_filename") or ""
                    extracted = extract_document_text(data, mime, filename)
                    max_len = 12000
                    if len(extracted) > max_len:
                        extracted = extracted[:max_len] + "\n[... documento truncado ...]"
                    text = f"O usuário enviou um documento. Conteúdo extraído:\n{extracted}"
                logger.info("webhook/whatsapp: mídia processada tipo=%s len=%d", text[:20], len(extracted))
            except Exception as e:
                logger.exception("webhook/whatsapp: falha ao processar mídia: %s", e)
    # Se ainda for marcador de mídia, não conseguimos obter/processar o arquivo → pedir texto
    media_reply: Optional[str] = None
    if text == "[AUDIO]":
        media_reply = "Não consegui acessar o áudio. Por favor, escreva sua dúvida ou pedido em uma mensagem de texto."
    elif text == "[IMAGE]":
        media_reply = "Não consegui acessar a imagem. Por favor, descreva em texto o que você precisa."
    elif text in ("[PDF]", "[EXCEL]", "[WORD]", "[DOCUMENT]"):
        media_reply = "Não consegui acessar o arquivo. Por favor, copie ou resuma as informações em uma mensagem de texto."
    if media_reply:
        logger.info("webhook/whatsapp: mídia sem acesso/processamento (%s), pedindo texto", text)
        send_whatsapp_message(phone, media_reply)
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

    # Extrair idade/turno/horário de estudo da mensagem e persistir na conversa
    try:
        info = extract_candidate_info(text)
        if info.get("candidate_age") is not None or info.get("study_shift") or info.get("study_hours"):
            update_candidate_info(
                phone,
                candidate_age=info.get("candidate_age"),
                study_shift=info.get("study_shift"),
                study_hours=info.get("study_hours"),
            )
    except Exception as e:
        logger.warning("webhook/whatsapp: falha ao extrair/atualizar candidate_info: %s", e)

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
