"""Teste de idempotência: processar duas vezes a mesma message_id não deve duplicar ação."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Requer banco configurado para rodar; skip se não houver
pytest.importorskip("pymysql")

from app.db import is_message_processed, mark_message_processed, get_connection


def test_mark_and_check_processed():
    msg_id = "test-idempotency-" + str(__import__("time").time())
    phone = "5585000000000"
    assert is_message_processed(msg_id) is False
    mark_message_processed(msg_id, phone)
    assert is_message_processed(msg_id) is True
    # Segunda vez ainda deve retornar True (já processada)
    assert is_message_processed(msg_id) is True
