"""Testes unitários da extração de idade e turno do candidato."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.candidate_info import extract_candidate_info


def test_17_anos_noturno_19h_22h():
    """(a) '17 anos, noturno 19h às 22h' -> age=17 (elegível), shift=noturno, hours=19:00-22:00"""
    r = extract_candidate_info("17 anos, noturno das 19h às 22h")
    assert r["candidate_age"] == 17
    assert r["study_shift"] == "noturno"
    assert r["study_hours"] == "19:00-22:00"


def test_15_anos_inelegivel():
    """(b) '15 anos' -> age=15 (inelegível)"""
    r = extract_candidate_info("15 anos")
    assert r["candidate_age"] == 15


def test_noturno_sem_idade_pedir_idade():
    """(c) 'noturno 19h às 22h' (sem idade) -> age=None (agente deve pedir idade)"""
    r = extract_candidate_info("noturno 19h às 22h")
    assert r["candidate_age"] is None
    assert r["study_shift"] == "noturno"
    assert r["study_hours"] == "19:00-22:00"


def test_17_isolado():
    """(d) '17' (isolado) -> age=17"""
    r = extract_candidate_info("17")
    assert r["candidate_age"] == 17


def test_dezessete_anos():
    """(e) 'dezessete anos' -> age=17"""
    r = extract_candidate_info("dezessete anos")
    assert r["candidate_age"] == 17


def test_17_anos_sem_confundir_com_horario():
    """Não confundir 19/22 de '19h às 22h' com idade."""
    r = extract_candidate_info("Tenho 17 anos, estudo noturno das 19h às 22h")
    assert r["candidate_age"] == 17
    assert r["study_shift"] == "noturno"
    assert r["study_hours"] == "19:00-22:00"


def test_tenho_17():
    r = extract_candidate_info("tenho 17")
    assert r["candidate_age"] == 17


def test_17a():
    r = extract_candidate_info("17a, noturno")
    assert r["candidate_age"] == 17
    assert r["study_shift"] == "noturno"


def test_faixa_plausivel_fora():
    r = extract_candidate_info("tenho 9 anos")
    assert r["candidate_age"] is None
    r = extract_candidate_info("85 anos")
    assert r["candidate_age"] is None
