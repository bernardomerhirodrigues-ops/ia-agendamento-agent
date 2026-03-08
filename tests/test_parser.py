"""Testes unitários do parser de mensagens e idempotência."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import _extract_whatsapp_payload


def test_extract_simple_payload():
    body = {"phone": "5585999999999", "text": "Quero agendar", "message_id": "msg123"}
    out = _extract_whatsapp_payload(body)
    assert out is not None
    assert out["phone"] == "5585999999999"
    assert out["text"] == "Quero agendar"
    assert out["message_id"] == "msg123"


def test_extract_with_from():
    body = {"from": "5585988888888", "body": "Oi", "message_id": "m1"}
    out = _extract_whatsapp_payload(body)
    assert out is not None
    assert out["phone"] == "5585988888888"
    assert out["text"] == "Oi"


def test_extract_meta_cloud_api():
    body = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{"id": "wamid.xyz", "from": "5511999999999", "text": {"body": "Sim"}}],
                    "contacts": [{"profile": {"first_name": "Maria"}}],
                }
            }]
        }]
    }
    out = _extract_whatsapp_payload(body)
    assert out is not None
    assert out["message_id"] == "wamid.xyz"
    assert out["phone"] == "5511999999999"
    assert out["text"] == "Sim"
    assert out["first_name"] == "Maria"


def test_extract_empty_returns_none():
    assert _extract_whatsapp_payload({}) is None
    assert _extract_whatsapp_payload({"other": "key"}) is None
