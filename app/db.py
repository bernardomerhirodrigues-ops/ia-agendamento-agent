import logging
import pymysql
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from .config import get_db_config

logger = logging.getLogger(__name__)


@contextmanager
def get_connection():
    cfg = get_db_config()
    conn = pymysql.connect(
        host=cfg["host"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset=cfg["charset"],
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def is_message_processed(message_id: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM ia_agendamento_processed_messages WHERE message_id = %s",
                (message_id,),
            )
            return cur.fetchone() is not None


def mark_message_processed(message_id: str, phone_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO ia_agendamento_processed_messages (message_id, phone_id) VALUES (%s, %s)",
                (message_id, phone_id),
            )


def try_mark_message_processed(message_id: str, phone_id: str) -> bool:
    """
    Tenta marcar mensagem como processada. Retorna True se foi a primeira (inseriu),
    False se já existia (evita duplicatas por race condition quando 2 webhooks chegam juntos).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO ia_agendamento_processed_messages (message_id, phone_id) VALUES (%s, %s)",
                (message_id, phone_id),
            )
            return cur.rowcount == 1


def get_conversation(phone_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, phone_id, first_name, flow_status, current_slot_date, current_slot_time, current_responsible, schedule_id FROM ia_agendamento_conversation WHERE phone_id = %s",
                (phone_id,),
            )
            return cur.fetchone()


def upsert_conversation(
    phone_id: str,
    first_name: Optional[str] = None,
    flow_status: Optional[str] = None,
    current_slot_date: Optional[str] = None,
    current_slot_time: Optional[str] = None,
    current_responsible: Optional[str] = None,
    schedule_id: Optional[int] = None,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ia_agendamento_conversation (phone_id, first_name, flow_status, current_slot_date, current_slot_time, current_responsible, schedule_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    updated_at = CURRENT_TIMESTAMP,
                    first_name = COALESCE(%s, first_name),
                    flow_status = COALESCE(%s, flow_status),
                    current_slot_date = COALESCE(%s, current_slot_date),
                    current_slot_time = COALESCE(%s, current_slot_time),
                    current_responsible = COALESCE(%s, current_responsible),
                    schedule_id = COALESCE(%s, schedule_id)
                """,
                (
                    phone_id,
                    first_name,
                    flow_status or "start",
                    current_slot_date,
                    current_slot_time,
                    current_responsible,
                    schedule_id,
                    first_name,
                    flow_status,
                    current_slot_date,
                    current_slot_time,
                    current_responsible,
                    schedule_id,
                ),
            )


def get_memory(phone_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM ia_agendamento_memory WHERE phone_id = %s ORDER BY created_at DESC LIMIT %s",
                (phone_id, limit),
            )
            rows = cur.fetchall()
    return list(reversed(rows))


def add_memory(phone_id: str, role: str, content: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ia_agendamento_memory (phone_id, role, content) VALUES (%s, %s, %s)",
                (phone_id, role, content[:16000]),
            )


def get_agent_config() -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT openai_api_key, openai_model, system_prompt, temperature, default_entrevistador, whatsapp_webhook_number, message_buffer_seconds FROM ia_agendamento_config WHERE enabled = 1 LIMIT 1"
            )
            return cur.fetchone()


def add_to_message_buffer(phone_id: str, message_id: str, text_content: str, first_name: str = "") -> bool:
    """Insere mensagem no buffer. Retorna True se inseriu, False se duplicata (message_id já existe)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO ia_agendamento_message_buffer (phone_id, message_id, text_content, first_name) VALUES (%s, %s, %s, %s)",
                (phone_id, message_id, (text_content or "")[:16000], (first_name or "")[:255]),
            )
            return cur.rowcount == 1


def get_buffered_messages(phone_id: str) -> List[Dict[str, Any]]:
    """Retorna mensagens do buffer para o phone, ordenadas por created_at."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT message_id, text_content, first_name, created_at FROM ia_agendamento_message_buffer WHERE phone_id = %s ORDER BY created_at ASC",
                (phone_id,),
            )
            return cur.fetchall()


def delete_buffer_for_phone(phone_id: str) -> None:
    """Remove todas as mensagens do buffer para o phone (após processar)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ia_agendamento_message_buffer WHERE phone_id = %s", (phone_id,))


def get_phones_with_buffer_older_than_seconds(seconds: int) -> List[str]:
    """Retorna phone_ids que têm mensagens no buffer com a mais antiga há mais de seconds segundos."""
    if seconds <= 0:
        return []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT phone_id FROM ia_agendamento_message_buffer GROUP BY phone_id "
                "HAVING MIN(created_at) < DATE_SUB(NOW(), INTERVAL %s SECOND)",
                (seconds,),
            )
            return [row["phone_id"] for row in cur.fetchall()]


def mark_human_takeover_chat(chat_id: str) -> None:
    """Registra que um humano assumiu este chat (ex.: mensagem enviada pelo negócio). O agente não responderá mais."""
    if not (chat_id or "").strip():
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO ia_agendamento_human_takeover (chat_id) VALUES (%s)",
                (chat_id.strip(),),
            )


def is_chat_human_takeover(chat_id: str) -> bool:
    """Retorna True se este chat foi marcado como assumido por humano."""
    if not (chat_id or "").strip():
        return False
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM ia_agendamento_human_takeover WHERE chat_id = %s LIMIT 1",
                (chat_id.strip(),),
            )
            return cur.fetchone() is not None
