import re
from pathlib import Path


FAULT_CODE_PATTERN = re.compile(
    r"(?im)^\s*((?:F|E|A)\d{3,5}|(?:fault|alarm|error)(?:\s+code)?\s*(?:0x)?[A-Z0-9-]+)\b"
)


def _contains(text: str, *terms: str) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def classify_document(file_name: str, text: str) -> str:
    evidence = f"{Path(file_name).stem}\n{text[:12000]}"
    fault_count = len(FAULT_CODE_PATTERN.findall(evidence))
    if fault_count >= 2 or _contains(
        evidence,
        "故障代码",
        "故障码",
        "fault code",
        "alarm code",
        "error code",
    ):
        return "fault_code"
    if _contains(evidence, "标准操作规程", "操作步骤", "operating procedure", "sop"):
        return "sop"
    if _contains(evidence, "维护手册", "点检", "保养周期", "maintenance schedule"):
        return "maintenance"
    if _contains(evidence, "技术规格", "technical specification", "技术参数"):
        return "technical_spec"
    if _contains(
        evidence,
        "操作说明",
        "使用说明",
        "用户手册",
        "operating instructions",
        "instruction manual",
        "user manual",
    ):
        return "manual"
    return "general"


def infer_language(text: str) -> str:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text[:12000]))
    latin = len(re.findall(r"[A-Za-z]", text[:12000]))
    if chinese and latin and min(chinese, latin) / max(chinese, latin) >= 0.2:
        return "mixed"
    if chinese:
        return "zh"
    if latin:
        return "en"
    return "unknown"


def infer_manufacturer(text: str) -> str:
    manufacturers = (
        ("Siemens", ("siemens", "西门子")),
        ("ABB", ("abb",)),
        ("Mitsubishi", ("mitsubishi", "三菱")),
        ("Rockwell Automation", ("rockwell", "allen-bradley")),
        ("Schneider Electric", ("schneider", "施耐德")),
        ("Omron", ("omron", "欧姆龙")),
    )
    for name, aliases in manufacturers:
        if _contains(text[:12000], *aliases):
            return name
    match = re.search(r"(?im)^\s*(?:制造商|manufacturer)\s*[:：]\s*(.{2,80})$", text)
    return match.group(1).strip() if match else ""


def infer_equipment_type(text: str) -> str:
    mappings = (
        ("PLC", ("plc", "可编程逻辑控制器")),
        ("variable_frequency_drive", ("变频器", "frequency converter", "variable frequency drive")),
        ("servo_drive", ("伺服驱动", "servo drive")),
        ("industrial_robot", ("工业机器人", "industrial robot")),
        ("sensor", ("传感器", "sensor")),
    )
    for value, terms in mappings:
        if _contains(text[:12000], *terms):
            return value
    return ""


def infer_equipment_model(text: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:设备型号|产品型号|型号|model)\s*[:：]\s*([A-Z0-9][A-Z0-9._/-]{2,40})\s*$",
        text,
    )
    return match.group(1).strip() if match else ""


def infer_title(text: str, file_name: str) -> str:
    for line in text.splitlines()[:40]:
        candidate = line.strip()
        if 3 <= len(candidate) <= 120 and not re.match(
            r"(?i)^(?:制造商|manufacturer|设备型号|model)\s*[:：]",
            candidate,
        ):
            return candidate
    return Path(file_name).stem


def infer_version(text: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:文档版本|版本|document version|revision|rev\.)\s*[:：]?\s*([A-Z0-9._-]{1,30})\s*$",
        text,
    )
    return match.group(1).strip() if match else ""


def infer_publish_date(text: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:发布日期|发布日|publish date)\s*[:：]?\s*(\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?)\s*$",
        text,
    )
    return match.group(1).strip() if match else ""


def classify_knowledge(section: str, content: str, error_code: str = "") -> str:
    evidence = f"{section}\n{content[:3000]}"
    if error_code or FAULT_CODE_PATTERN.search(content):
        return "fault"
    if _contains(evidence, "警告", "危险", "注意事项", "warning", "caution", "danger"):
        return "warning"
    if _contains(
        evidence,
        "操作步骤",
        "启动流程",
        "准备工作",
        "操作规程",
        "procedure",
        "步骤",
    ):
        return "procedure"
    if _contains(evidence, "维护", "保养", "点检", "更换周期", "maintenance"):
        return "maintenance"
    parameter_lines = re.findall(
        r"(?im)^\s*[^\n:：]{1,50}[:：]\s*[-+]?\d+(?:\.\d+)?\s*(?:v|a|kw|w|hz|℃|°c|mm|ms|s|rpm|%)?\b",
        content,
    )
    if len(parameter_lines) >= 2:
        return "parameter"
    if _contains(evidence, "技术规格", "specification", "技术参数"):
        return "specification"
    if _contains(evidence, "概述", "简介", "overview"):
        return "overview"
    return "general"
