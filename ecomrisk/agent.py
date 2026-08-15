"""Risk review agent: ReAct loop, tools, and a deterministic offline policy."""

from __future__ import annotations

import json
import os
import urllib.request

import pandas as pd

from .data import RULES


class ToolRegistry:
    """Read-only business tools the agent can call. Kept separate from the model for testability."""

    def __init__(self, report: pd.DataFrame, merchants: pd.DataFrame, products: pd.DataFrame):
        self.report = report.set_index("merchant_id", drop=False)
        self.merchants = merchants.set_index("merchant_id", drop=False)
        self.products = products
        self.rules = {r["id"]: r for r in RULES}

    def query_risk_report(self, merchant_id: str) -> dict:
        if merchant_id not in self.report.index:
            return {"merchant_id": merchant_id, "risk_score": 0.0, "predicted_risk": 0, "evidence": "未命中风险"}
        row = self.report.loc[merchant_id]
        return {
            "merchant_id": merchant_id,
            "risk_score": round(float(row["risk_score"]), 3),
            "predicted_risk": int(row["predicted_risk"]),
            "evidence": str(row["evidence"]),
            "true_types": str(row["true_types"]),
        }

    def lookup_product(self, merchant_id: str) -> dict:
        sample = self.products[self.products["merchant_id"] == merchant_id].head(1)
        if sample.empty:
            return {"merchant_id": merchant_id, "sample": None}
        s = sample.iloc[0]
        return {"merchant_id": merchant_id, "sample": {"title": s["title"], "price": s["price"], "category": s["category"]}}

    def search_rules(self, query: str) -> list[dict]:
        q = query or ""
        return [r for r in RULES if r["title"] in q or r["title"][:2] in q or q in r["title"]]

    def create_case(self, merchant_id: str, action: str, reason: str) -> dict:
        return {"case_id": f"CASE-{abs(hash(merchant_id + action)) % 100000:05d}", "merchant_id": merchant_id, "action": action, "reason": reason}


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None, temperature: float = 0.1):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def chat(self, messages: list[dict]) -> dict:
        url = self.base_url + "/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


SYSTEM_PROMPT = (
    "你是电商风控研判 Agent。你只能通过给定工具获取信息，然后给出结构化结论。"
    "每一步只返回一个 JSON：要么调用一个工具 {\"tool\": \"...\", \"tool_input\": {...}}，"
    "要么给出最终结论 {\"final\": {\"merchant_id\":..., \"risk_level\":..., \"violation_types\":[...], "
    "\"confidence\":0.0, \"evidence\":[...], \"recommended_action\":\"...\", \"explanation\":\"...\"}}。"
    "可用工具：query_risk_report, lookup_product, search_rules, create_case。"
)


class DeterministicPolicy:
    """Offline, fully deterministic reviewer that walks the same tool sequence."""

    def review(self, tools: ToolRegistry, merchant_id: str) -> dict:
        report = tools.query_risk_report(merchant_id)
        first = (report["evidence"].split("；")[0] if report["evidence"] and report["evidence"] != "无明显异常" else "")
        rules = tools.search_rules(first) if first else []
        product = tools.lookup_product(merchant_id)

        violations = [t for t in str(report["true_types"]).split("|") if t] if report["true_types"] else []
        if not violations and first:
            violations = [first]

        score = report["risk_score"]
        if score >= 0.7:
            level, action = "high", "冻结店铺并升级人审"
        elif score >= 0.5:
            level, action = "medium", "下架风险商品，限期整改"
        else:
            level, action = "low", "通过，进入常规监控"

        case = tools.create_case(merchant_id, action, ";".join(violations or ["低风险复核"]))
        return {
            "merchant_id": merchant_id,
            "risk_level": level,
            "violation_types": violations or [],
            "confidence": round(min(0.95, max(0.5, score)), 2),
            "evidence": report["evidence"].split("；") if report["evidence"] != "无明显异常" else [],
            "recommended_action": action,
            "explanation": f"风险分为 {report['risk_score']}，命中证据：{report['evidence']}。匹配规则：{';'.join(r['id'] for r in rules) or '无'}。商品样本：{product.get('sample')}。已生成处置工单 {case['case_id']}。",
        }


def run_react(llm: OpenAICompatLLM, tools: ToolRegistry, merchant_id: str, max_steps: int = 6) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"研判商家 {merchant_id}"}]
    for _ in range(max_steps):
        action = llm.chat(messages)
        if "tool" in action:
            name = action["tool"]
            tool_input = action.get("tool_input", {})
            method = getattr(tools, name, None)
            if method is None:
                observation = {"error": f"unknown tool {name}"}
            else:
                observation = method(**tool_input)
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            messages.append({"role": "user", "content": json.dumps(observation, ensure_ascii=False)})
            continue
        if "final" in action:
            return action["final"]
    raise RuntimeError("agent exceeded max steps without a final answer")
