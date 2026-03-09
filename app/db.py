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
                "SELECT openai_api_key, openai_model, system_prompt, temperature, default_entrevistador, whatsapp_webhook_number FROM ia_agendamento_config WHERE enabled = 1 LIMIT 1"
            )
            return cur.fetchone()
