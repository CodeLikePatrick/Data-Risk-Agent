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
    # 地址与地理
    {"id": "ADDR-01", "title": "地址真实性", "condition": "地址无法解析到行政区，或为虚构门牌号", "action": "冻结店铺并限期补充资质", "why": "虚假地址常用于注册空壳店铺、规避属地监管"},
    {"id": "ADDR-02", "title": "地址与坐标矛盾", "condition": "声称的城市或区县与经纬度不一致", "action": "标记风险，要求更新地址", "why": "坐标矛盾说明地址信息不实或使用了第三方乱填数据"},
    {"id": "ADDR-03", "title": "门址或楼栋无法定位", "condition": "门牌号、楼栋号无法映射到真实 POI", "action": "下架受影响商品并补充定位信息", "why": "无法定位的门址会导致配送失败和地址绑定率下降"},
    {"id": "ADDR-04", "title": "多店铺共用地址", "condition": "多个店铺使用同一地址或同一联系人", "action": "升级人审，纳入团伙排查", "why": "共用地址是团伙经营和刷单的常见特征"},
    {"id": "ADDR-05", "title": "地址跨区域跳变", "condition": "同一商家地址在短时间内跨城市或跨省份变化", "action": "冻结并核查经营主体", "why": "频繁跳变可能涉及账号买卖或黑灰产转移"},
    # 价格
    {"id": "PRICE-01", "title": "异常低价", "condition": "价格明显低于同品类成本线或中位数", "action": "下架商品并标记价格风险", "why": "异常低价常伴随售假、引流、刷单"},
    {"id": "PRICE-02", "title": "价格偏离品类", "condition": "价格偏离同品类中位数 3 倍标准差以上", "action": "进入价格审核队列", "why": "显著偏离说明定价异常或类目错放"},
    {"id": "PRICE-03", "title": "价格先涨后降", "condition": "短期内先提价再大幅降价，制造虚假优惠", "action": "下架并警告虚假促销", "why": "虚假折扣误导消费者，属于常见违规"},
    {"id": "PRICE-04", "title": "超高价异常", "condition": "价格远高于同类商品且无合理依据", "action": "标记疑似虚假宣传并进入人工审核", "why": "超高价可能变相售假或进行价格欺诈"},
    # 文本与内容
    {"id": "TEXT-01", "title": "标题关键词堆砌", "condition": "标题堆砌品牌词、品类词或违禁词", "action": "下架并责令改写", "why": "关键词堆砌影响搜索质量与用户体验"},
    {"id": "TEXT-02", "title": "标题模板化", "condition": "大量标题由同一模板生成、字符信息熵过低", "action": "批量下架并整改", "why": "模板化标题常由黑灰产批量发布"},
    {"id": "TEXT-03", "title": "标题与商品不符", "condition": "标题描述与商品图片或属性不一致", "action": "下架并核实商品信息", "why": "货不对板属于虚假宣传"},
    {"id": "TEXT-04", "title": "评价文案异常", "condition": "同一店铺或不同店铺出现批量雷同评价", "action": "删除异常评价并计入信用", "why": "刷好评会扭曲平台信用体系"},
    # 账号与团伙
    {"id": "GROUP-01", "title": "共享设备", "condition": "多个账号共用设备指纹", "action": "升级人审并关联账号", "why": "共享设备是团伙刷单、操纵评分的强信号"},
    {"id": "GROUP-02", "title": "共享地址或联系人", "condition": "多个店铺共用地址、手机号或联系人", "action": "纳入团伙处置", "why": "共享联系信息说明背后是同一运营主体"},
    {"id": "GROUP-03", "title": "集中注册与集中上线", "condition": "大量账号在短时间集中注册并上线相似商品", "action": "批量冻结并审查", "why": "集中注册上线是黑产批量作恶的典型模式"},
    {"id": "GROUP-04", "title": "团伙聚集", "condition": "地址、设备、行为等多个维度同时聚集", "action": "纳入团伙处置并联合排查", "why": "多维度聚集可提升团伙识别的置信度"},
    # 交易与订单
    {"id": "ORDER-01", "title": "刷单行为", "condition": "下单-取消比异常，或订单集中由少数账号产生", "action": "标记订单异常并降权", "why": "刷单会破坏排名与评价的公信力"},
    {"id": "ORDER-02", "title": "退款与纠纷异常", "condition": "退款率、纠纷率显著高于同品类", "action": "进入质量与信用复核", "why": "异常退款说明商品或服务存在系统性问题"},
    {"id": "ORDER-03", "title": "发货地址与商家地址不符", "condition": "实际发货地址与注册地址或承诺地址不一致", "action": "核查货源并标记风险", "why": "发货地址不符可能涉及代发、售假或跨区经营"},
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
