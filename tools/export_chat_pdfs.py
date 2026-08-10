from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
)
from xml.sax.saxutils import escape


TITLE_BY_ID = {
    "019feb01-054a-7841-8354-18cfe630e6e4": "查找聊天记录下载方法",
    "019fd71c-1225-7b73-819f-296e37bce246": "审阅毕业论文LaTeX",
    "019fd10e-8447-7f32-87b7-23e7f160e8dc": "解析论文 LaTeX 与 Bib 引用",
    "019fad06-3107-7fb2-a811-d5e06fc9dee3": "抽取 Wednesday 流量与数据包外部测试集",
    "019f8db3-bffc-7ff0-af5d-04a1d37d1d79": "run_stage1.py 运行结果详细解析",
    "019f6f18-06a4-7c80-9656-4eec3e99512a": "修复 Stage2Dataset 参数错误",
    "019f4b03-e09c-7110-a425-04e6d87dec5f": "梳理论文补充要求",
    "019f461f-28ef-7b93-96d2-f67e52cae570": "解析 s2 代码",
    "019f3b5f-5533-7760-8231-bfe776cc71c3": "分析 Stage2 结果",
    "019f2244-6a3e-7f51-853d-fa086f732b61": "记录想法",
    "019f223c-9782-7d11-a1b3-29646d23c4b9": "记录我的idea",
    "019f2226-cb6a-7ad3-8378-8dcaf666fffe": "Review open PRs",
}


def register_fonts() -> tuple[str, str]:
    candidates = [
        (Path(r"C:\Windows\Fonts\msyh.ttc"), "MicrosoftYaHei"),
        (Path(r"C:\Windows\Fonts\simhei.ttf"), "SimHei"),
        (Path(r"C:\Windows\Fonts\simsun.ttc"), "SimSun"),
    ]
    chosen = None
    for path, name in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path), subfontIndex=0))
            chosen = name
            break
    if not chosen:
        raise RuntimeError("No usable Chinese font found in C:\\Windows\\Fonts")
    mono_path = Path(r"C:\Windows\Fonts\consola.ttf")
    mono = chosen
    if mono_path.exists():
        pdfmetrics.registerFont(TTFont("Consolas", str(mono_path)))
        mono = "Consolas"
    return chosen, mono


def clean_filename(value: str, limit: int = 100) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().rstrip(".")
    value = re.sub(r"\s+", " ", value)
    return (value[:limit].rstrip() or "未命名聊天") + ".pdf"


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_text", "output_text", "text"}:
            out.append(part.get("text", ""))
    return "\n".join(x for x in out if x)


def clean_user_text(text: str) -> str:
    for tag in ("recommended_plugins", "environment_context"):
        text = re.sub(rf"<{tag}>[\s\S]*?</{tag}>", "", text, flags=re.IGNORECASE)
    if text.lstrip().startswith("The following is the Codex agent history whose request action you are assessing"):
        return ""
    return text.strip()


def parse_jsonl(path: Path):
    session_id = path.stem.rsplit("-", 5)[-5:]
    session_id = "-".join(session_id)
    messages = []
    first_timestamp = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("timestamp")
            if first_timestamp is None and ts:
                first_timestamp = ts
            payload = row.get("payload") or {}
            if row.get("type") != "response_item" or payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = content_text(payload.get("content"))
            if role == "user":
                text = clean_user_text(text)
            if text.strip():
                messages.append({"role": role, "text": text, "timestamp": ts})
    # The UUID is always present verbatim in the filename.
    match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", path.name)
    if match:
        session_id = match.group(1)
    return session_id, first_timestamp, messages


def split_blocks(text: str):
    parts = re.split(r"(```[\s\S]*?```)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("```") and part.endswith("```"):
            lines = part.splitlines()
            yield "code", "\n".join(lines[1:-1])
        else:
            for para in re.split(r"\n\s*\n", part):
                if para.strip():
                    yield "text", para.strip()


def format_ts(value: str | None) -> str:
    if not value:
        return ""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def build_pdf(out_path: Path, title: str, session_id: str, started: str | None, messages, body_font: str, mono_font: str):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ChatTitle", parent=styles["Title"], fontName=body_font, fontSize=19, leading=26, alignment=TA_CENTER, textColor=colors.HexColor("#172B4D"), spaceAfter=8)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontName=body_font, fontSize=8.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#6B778C"), spaceAfter=16)
    role_styles = {
        "user": ParagraphStyle("UserRole", parent=styles["Heading3"], fontName=body_font, fontSize=11, leading=15, textColor=colors.HexColor("#0052CC"), spaceBefore=7, spaceAfter=4),
        "assistant": ParagraphStyle("AssistantRole", parent=styles["Heading3"], fontName=body_font, fontSize=11, leading=15, textColor=colors.HexColor("#006644"), spaceBefore=7, spaceAfter=4),
    }
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontName=body_font, fontSize=9.5, leading=15, alignment=TA_LEFT, textColor=colors.HexColor("#172B4D"), spaceAfter=7, splitLongWords=True)
    code_style = ParagraphStyle("Code", parent=styles["Code"], fontName=mono_font, fontSize=7.5, leading=10.5, leftIndent=6, rightIndent=6, borderColor=colors.HexColor("#DFE1E6"), borderWidth=0.5, borderPadding=6, backColor=colors.HexColor("#F4F5F7"), spaceAfter=8, splitLongWords=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(body_font, 8)
        canvas.setFillColor(colors.HexColor("#6B778C"))
        canvas.drawString(18 * mm, 10 * mm, title[:48])
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"第 {doc.page} 页")
        canvas.restoreState()

    doc = BaseDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=17 * mm, title=title, author="Codex Chat Export")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="chat", frames=frame, onPage=footer)])
    story = [Paragraph(escape(title), title_style), Paragraph(escape(f"会话 ID: {session_id}    开始时间: {format_ts(started)}    消息数: {len(messages)}"), meta_style)]
    for msg in messages:
        role = msg["role"]
        label = "用户" if role == "user" else "助手"
        stamp = format_ts(msg.get("timestamp"))
        story.append(Paragraph(escape(f"{label}  {stamp}"), role_styles[role]))
        for kind, block in split_blocks(msg["text"]):
            if kind == "code":
                story.append(Preformatted(block, code_style, maxLineLength=110))
            else:
                html = escape(block).replace("\n", "<br/>")
                story.append(Paragraph(html, body_style))
        story.append(Spacer(1, 3 * mm))
    if not messages:
        story.append(Paragraph("未发现可导出的用户或助手消息。", body_style))
    doc.build(story)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-json", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    body_font, mono_font = register_fonts()
    files = sorted(set(args.sessions.rglob("*.jsonl")) | set(args.archived.rglob("*.jsonl")))
    used = {}
    manifest = []
    for src in files:
        session_id, started, messages = parse_jsonl(src)
        if session_id not in TITLE_BY_ID:
            continue
        fallback = next(
            (
                m["text"].strip().splitlines()[0][:80]
                for m in messages
                if m["role"] == "user"
                and not m["text"].lstrip().startswith(("<environment_context>", "<recommended_plugins>"))
                and not m["text"].lstrip().startswith("The following is the Codex agent history")
            ),
            src.stem,
        )
        title = TITLE_BY_ID.get(session_id, fallback)
        base = clean_filename(title)
        count = used.get(base, 0) + 1
        used[base] = count
        filename = base if count == 1 else base[:-4] + f" ({count}).pdf"
        out = args.output / filename
        build_pdf(out, title, session_id, started, messages, body_font, mono_font)
        manifest.append({"title": title, "session_id": session_id, "messages": len(messages), "source": str(src), "pdf": str(out)})
        print(f"{out.name}\t{len(messages)} messages")
    if args.extra_json and args.extra_json.exists():
        extra = json.loads(args.extra_json.read_text(encoding="utf-8"))
        title = extra["title"]
        out = args.output / clean_filename(title)
        build_pdf(out, title, extra["session_id"], extra.get("started"), extra["messages"], body_font, mono_font)
        manifest.append({"title": title, "session_id": extra["session_id"], "messages": len(extra["messages"]), "source": str(args.extra_json), "pdf": str(out)})
        print(f"{out.name}\t{len(extra['messages'])} messages")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL\t{len(manifest)} PDFs")


if __name__ == "__main__":
    main()
