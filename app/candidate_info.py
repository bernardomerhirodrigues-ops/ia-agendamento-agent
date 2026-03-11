"""
Extração robusta de idade e turno/horário de estudo a partir da mensagem do candidato.
Prioriza números no contexto de idade (anos, tenho) e evita confundir com horários (19h, 22h).
"""
import re
from typing import Optional, Dict, Any

# Faixa plausível para idade (anos)
AGE_MIN = 10
AGE_MAX = 80

# Números por extenso comuns (minúsculo, sem acento para match)
_IDADE_EXTENSO = {
    "dez": 10, "onze": 11, "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14,
    "quinze": 15, "dezesseis": 16, "dezessete": 17, "dezoito": 18, "dezenove": 19,
    "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50,
    "sessenta": 60, "setenta": 70, "oitenta": 80,
}


def _normalize_for_match(text: str) -> str:
    """Remove acentos e normaliza para match de palavras."""
    if not text:
        return ""
    t = text.lower().strip()
    replacements = (
        ("á", "a"), ("à", "a"), ("ã", "a"), ("â", "a"),
        ("é", "e"), ("ê", "e"), ("í", "i"), ("ó", "o"), ("ô", "o"), ("õ", "o"), ("ú", "u"), ("ç", "c"),
    )
    for a, b in replacements:
        t = t.replace(a, b)
    return t


def _extract_age(text: str) -> Optional[int]:
    """
    Extrai idade em anos. Prioriza contexto: "X anos", "tenho X", "idade X".
    Ignora números que claramente são horário (ex.: 19h, 22h, 19:00).
    """
    if not text or not text.strip():
        return None
    normalized = _normalize_for_match(text)
    # Remover padrões de horário para não confundir: 19h, 22h, 19:00, 19h às 22h
    text_sem_hora = re.sub(r"\d{1,2}\s*:\s*\d{2}", " ", text)
    text_sem_hora = re.sub(r"\d{1,2}\s*h\s*(?:às|\-|a|\s*\d)", " ", text_sem_hora, flags=re.I)
    normalized_sem_hora = _normalize_for_match(text_sem_hora)

    # 1) Por extenso: "dezessete anos", "tenho dezessete"
    for word, num in _IDADE_EXTENSO.items():
        if word in normalized_sem_hora and (
            "ano" in normalized_sem_hora or "idade" in normalized_sem_hora or "tenho" in normalized_sem_hora
            or word + " " in normalized_sem_hora or " " + word in normalized_sem_hora
        ):
            return num if AGE_MIN <= num <= AGE_MAX else None

    # 2) Padrões explícitos de idade (prioridade alta)
    # "17 anos", "17 anos,", "tenho 17", "tenho 17 anos", "17 anos de idade", "idade 17", "faço 17"
    patterns_idade = [
        r"(?:tenho|idade|faço|com)\s*(\d{1,3})\s*(?:anos?)?",
        r"(\d{1,3})\s*anos?(?:\s+de\s+idade)?(?:\s|,|\.|$)",
        r"(\d{1,3})\s*a\b(?:\s|,|\.|$)",   # "17 a" ou "17 a,"
        r"(\d{1,3})a\b(?:\s|,|\.|$)",      # "17a," "17a."
        r"(\d{1,3})\s*anos?\s*(?:,|\.|$)",
    ]
    for pat in patterns_idade:
        m = re.search(pat, normalized_sem_hora, re.I)
        if m:
            try:
                age = int(m.group(1))
                if AGE_MIN <= age <= AGE_MAX:
                    return age
            except (ValueError, IndexError):
                pass

    # 3) Número isolado no início ou após vírgula (ex.: "17", "17, noturno") – só se não parecer horário
    for m in re.finditer(r"(?:^|[\s,])(\d{1,2})(?:[\s,]|$)", normalized_sem_hora):
        try:
            num_str = m.group(1)
            age = int(num_str)
            if age < AGE_MIN or age > AGE_MAX:
                continue
            # Se no texto original o número estiver colado a "h" ou ":", é horário (ex.: 19h, 22h)
            idx = text_sem_hora.find(num_str)
            if idx >= 0:
                after = text_sem_hora[idx + len(num_str) : idx + len(num_str) + 2].lower()
                before = text_sem_hora[max(0, idx - 2) : idx].lower()
                if after.startswith("h") or ":" in after or ":" in before:
                    continue
            return age
        except (ValueError, IndexError):
            continue

    return None


def _extract_study_shift(text: str) -> Optional[str]:
    """Extrai turno de estudo: noturno, noite, manhã, tarde, integral."""
    if not text or not text.strip():
        return None
    normalized = _normalize_for_match(text)
    if re.search(r"\bnoturno\b", normalized):
        return "noturno"
    if re.search(r"\bnoite\b|\bperiodo\s+da\s+noite\b|\bà noite\b", normalized):
        return "noturno"
    if re.search(r"\bmanha\b|\bmatutino\b|\bperiodo\s+da\s+manha\b|\bde manha\b", normalized):
        return "manhã"
    if re.search(r"\btarde\b|\bvespertino\b|\bperiodo\s+da\s+tarde\b|\bà tarde\b", normalized):
        return "tarde"
    if re.search(r"\bintegral\b|\bperiodo\s+integral\b", normalized):
        return "integral"
    return None


def _parse_time_part(s: str) -> Optional[str]:
    """Converte '19h', '19:00', '19' em '19:00'."""
    s = (s or "").strip()
    m = re.match(r"(\d{1,2})\s*:\s*(\d{2})", s)
    if m:
        h, mmm = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mmm <= 59:
            return f"{h:02d}:{mmm:02d}"
    m = re.match(r"(\d{1,2})\s*h(?:\s*(\d{2}))?", s, re.I)
    if m:
        h = int(m.group(1))
        mmm = int(m.group(2) or 0)
        if 0 <= h <= 23 and 0 <= mmm <= 59:
            return f"{h:02d}:{mmm:02d}"
    m = re.match(r"^(\d{1,2})$", s)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return f"{h:02d}:00"
    return None


def _extract_study_hours(text: str) -> Optional[str]:
    """Extrai faixa de horário de estudo: '19h às 22h' -> '19:00-22:00'."""
    if not text or not text.strip():
        return None
    # Padrões: "19h às 22h", "das 19h às 22h", "19:00 às 22:00", "19h-22h", "19h as 22h"
    m = re.search(
        r"(?:das?\s+)?(\d{1,2}(?:\s*[h:]\s*\d{2})?)\s*(?:às|as|a|-)\s*(\d{1,2}(?:\s*[h:]\s*\d{2})?)",
        text,
        re.I,
    )
    if m:
        start = _parse_time_part(re.sub(r"\s+", "", m.group(1)))
        end = _parse_time_part(re.sub(r"\s+", "", m.group(2)))
        if start and end:
            return f"{start}-{end}"
    return None


def extract_candidate_info(text: str) -> Dict[str, Any]:
    """
    Extrai idade, turno e horário de estudo da mensagem.
    Retorna dict com: candidate_age (int ou None), study_shift (str ou None), study_hours (str ou None).
    Idade fora da faixa plausível (10-80) retorna None em candidate_age.
    """
    result: Dict[str, Any] = {
        "candidate_age": None,
        "study_shift": None,
        "study_hours": None,
    }
    if not text or not isinstance(text, str):
        return result
    t = text.strip()
    if not t:
        return result

    age = _extract_age(t)
    if age is not None:
        result["candidate_age"] = age
    result["study_shift"] = _extract_study_shift(t)
    result["study_hours"] = _extract_study_hours(t)
    return result
