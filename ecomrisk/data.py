"""Synthetic e-commerce data with injected risks, plus a rule knowledge base."""

from __future__ import annotations

import numpy as np
import pandas as pd


# District centers (lat, lon) used to generate plausible addresses.
DISTRICTS = {
    "上海市": {
        "浦东新区": (31.221, 121.541),
        "徐汇区": (31.188, 121.436),
        "闵行区": (31.112, 121.381),
    },
    "杭州市": {
        "西湖区": (30.259, 120.130),
        "余杭区": (30.420, 119.981),
    },
    "广州市": {
        "天河区": (23.125, 113.361),
        "白云区": (23.170, 113.273),
    },
}

CATEGORY_PRICE = {
    "数码": 1800.0,
    "服饰": 120.0,
    "食品": 35.0,
    "美妆": 150.0,
    "家居": 260.0,
}

# Risk rules. These double as the agent's retrievable knowledge base.
RULES = [
    {
        "id": "ADDR-01",
        "title": "地址真实性",
        "condition": "地址无法解析到行政区，或城市/区县与坐标不一致",
        "action": "冻结店铺，要求补充资质",
        "why": "虚假地址常用于注册空壳店铺、规避属地监管",
    },
    {
        "id": "PRICE-02",
        "title": "价格异常",
        "condition": "商品价格偏离同品类中位数 3 倍标准差以上，或明显低于成本线",
        "action": "下架商品，标记价格风险",
        "why": "异常低价常伴随售假、引流、刷单",
    },
    {
        "id": "TEXT-03",
        "title": "文本灌水",
        "condition": "标题模板化、关键词堆砌、信息熵过低",
        "action": "下架商品，进入文本质量整改",
        "why": "灌水标题影响搜索与用户体验，常被黑灰产批量使用",
    },
    {
        "id": "GROUP-04",
        "title": "团伙聚集",
        "condition": "多个店铺共享设备、IP、地址或联系人",
        "action": "升级人审，纳入团伙处置",
        "why": "设备/地址聚集是刷单和黑产团伙的典型特征",
    },
]


def _address(city: str, district: str, center: tuple[float, float], fake: bool, rng: np.random.Generator) -> dict:
    lat, lon = center
    if fake:
        # Put coordinates in a completely different region (far away).
        lat += rng.uniform(1.5, 3.0) * rng.choice([-1.0, 1.0])
        lon += rng.uniform(1.5, 3.0) * rng.choice([-1.0, 1.0])
    else:
        lat += rng.normal(0.0, 0.008)
        lon += rng.normal(0.0, 0.008)
    street = f"{rng.integers(1, 2000)}号"
    raw = f"{city}{district}{street}" if not fake else f"{city}{district}不存在街道{rng.integers(1, 999)}号"
    return {"raw": raw, "city": city, "district": district, "lat": round(lat, 5), "lon": round(lon, 5), "fake": fake}


def generate(seed: int = 42, n_merchants: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate merchants and products with known risk labels for evaluation."""
    rng = np.random.default_rng(seed)
    cities = list(DISTRICTS)

    merchants = []
    for mid in range(n_merchants):
        city = rng.choice(cities)
        district = rng.choice(list(DISTRICTS[city]))
        center = DISTRICTS[city][district]
        category = rng.choice(list(CATEGORY_PRICE))

        is_risk = rng.random() < 0.30
        types = []
        if is_risk:
            # Inject one or more violation types.
            if rng.random() < 0.45:
                types.append("地址真实性")
            if rng.random() < 0.45:
                types.append("价格异常")
            if rng.random() < 0.45:
                types.append("文本灌水")
            if rng.random() < 0.40:
                types.append("团伙聚集")
            if not types:
                types.append("价格异常")

        fake_addr = "地址真实性" in types
        address = _address(city, district, center, fake=fake_addr, rng=rng)

        # A few merchants share device group ids to mimic organized groups.
        device_group = None
        if "团伙聚集" in types:
            device_group = int(mid // 8)

        merchants.append({
            "merchant_id": f"M{mid:04d}",
            "name": f"商家{mid:04d}",
            "category": category,
            "city": city,
            "district": district,
            "address_raw": address["raw"],
            "lat": address["lat"],
            "lon": address["lon"],
            "address_fake": fake_addr,
            "device_group": device_group,
            "true_types": "|".join(types),
            "true_risk": is_risk,
        })

    merchants = pd.DataFrame(merchants)

    products = []
    for _, m in merchants.iterrows():
        n_prod = int(rng.integers(3, 9))
        base = CATEGORY_PRICE[m["category"]]
        for _ in range(n_prod):
            spam = ("文本灌水" in m["true_types"].split("|")) and rng.random() < 0.7
            if spam:
                title = (f"{m['category']}爆款 {m['category']}正品 特价包邮 {m['category']}限时")
            else:
                title = f"{m['category']} {rng.choice(['新款', '经典', '精选', '简约'])}{rng.integers(10, 99)}"

            price_anomaly = ("价格异常" in m["true_types"].split("|")) and rng.random() < 0.7
            if price_anomaly:
                price = base * rng.choice([0.015, 0.02, 8.0, 10.0])
            else:
                price = base * rng.lognormal(0.0, 0.25)

            products.append({
                "product_id": f"P{len(products):05d}",
                "merchant_id": m["merchant_id"],
                "category": m["category"],
                "title": title,
                "price": round(float(price), 2),
            })

    return merchants, pd.DataFrame(products)
