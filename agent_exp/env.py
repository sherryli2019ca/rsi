"""ShopDesk: a deterministic customer-service tool environment.

Every tool is a pure function of a seeded database, so recorded tool results
can be replayed exactly and any prefix of an episode can be resumed.
Answers are checked exactly, so success needs no judge.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass


@dataclass
class Task:
    tid: str
    question: str
    answer: str
    kind: str


class ShopDB:
    def __init__(self, seed: int = 0, n_customers=30, n_products=25):
        rng = random.Random(seed)
        cats = ["electronics", "books", "home", "toys"]
        self.products = {
            f"P{p:03d}": {"sku": f"P{p:03d}", "name": f"Item {p}",
                          "category": rng.choice(cats),
                          "price_cents": rng.randrange(500, 40000, 5)}
            for p in range(n_products)}
        self.customers, self.orders = {}, {}
        oid = 1000
        for c in range(n_customers):
            cid = f"C{c:03d}"
            self.customers[cid] = {"customer_id": cid, "name": f"Customer {c}",
                                   "tier": "gold" if rng.random() < 0.3 else "basic"}
            for _ in range(rng.randint(1, 7)):
                items = [(rng.choice(list(self.products)), rng.randint(1, 3))
                         for _ in range(rng.randint(1, 3))]
                self.orders[f"O{oid}"] = {
                    "order_id": f"O{oid}", "customer_id": cid,
                    "items": [{"sku": s, "qty": q} for s, q in items],
                    "status": rng.choice(["delivered", "delivered", "shipped", "cancelled"]),
                    "days_since_delivery": rng.randint(1, 90)}
                oid += 1
        self.policies = {
            "refund": ("Delivered orders can be refunded within 30 days of delivery; "
                       "gold-tier customers have 60 days. Electronics carry a 10% restocking "
                       "fee unless the customer is gold tier. Refunds cover item prices only."),
            "cancellation": "Only shipped orders can be cancelled.",
        }

    # ---- ground truth helpers --------------------------------------------
    def order_total_cents(self, oid):
        return sum(self.products[i["sku"]]["price_cents"] * i["qty"]
                   for i in self.orders[oid]["items"])

    def refund_cents(self, oid):
        o = self.orders[oid]
        gold = self.customers[o["customer_id"]]["tier"] == "gold"
        window = 60 if gold else 30
        if o["status"] != "delivered" or o["days_since_delivery"] > window:
            return 0
        total = 0.0
        for it in o["items"]:
            p = self.products[it["sku"]]
            amt = p["price_cents"] * it["qty"]
            if p["category"] == "electronics" and not gold:
                amt *= 0.9
            total += amt
        return round(total)

    def customer_orders(self, cid):
        return sorted(k for k, o in self.orders.items() if o["customer_id"] == cid)


def money(cents) -> str:
    return f"{cents / 100:.2f}"


class Tools:
    """Tool implementations. `page_size` is part of the list_orders component."""

    def __init__(self, db: ShopDB, page_size: int = 3):
        self.db, self.page_size = db, page_size

    def call(self, name: str, args: dict) -> str:
        try:
            return json.dumps(getattr(self, name)(**args))
        except Exception as e:  # tool errors are returned to the agent
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def get_customer(self, customer_id):
        return self.db.customers[customer_id]

    def list_orders(self, customer_id, page=1):
        ids = self.db.customer_orders(customer_id)
        s = (int(page) - 1) * self.page_size
        return {"order_ids": ids[s:s + self.page_size], "page": int(page),
                "has_more": s + self.page_size < len(ids)}

    def get_order(self, order_id):
        return self.db.orders[order_id]

    def get_product(self, sku):
        return self.db.products[sku]

    def get_policy(self, name):
        return {"policy": self.db.policies[name]}

    def calculate(self, expression):
        if not set(expression) <= set("0123456789+-*/(). "):
            raise ValueError("only arithmetic is allowed")
        return {"result": eval(expression, {"__builtins__": {}}, {})}


def make_tasks(db: ShopDB, n: int, seed: int = 1) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    custs = list(db.customers)
    delivered = [k for k, o in db.orders.items() if o["status"] == "delivered"]
    for t in range(n):
        kind = rng.choice(["refund", "refund", "count", "spend"])
        if kind == "refund":
            oid = rng.choice(delivered)
            q = (f"Customer {db.orders[oid]['customer_id']} asks for a refund of order {oid}. "
                 f"How much should be refunded, in dollars?")
            a = money(db.refund_cents(oid))
        elif kind == "count":
            cid = rng.choice(custs)
            q = f"How many orders has customer {cid} placed in total?"
            a = str(len(db.customer_orders(cid)))
        else:
            cid = rng.choice(custs)
            ids = [o for o in db.customer_orders(cid) if db.orders[o]["status"] != "cancelled"]
            q = (f"What is the total value, in dollars, of all non-cancelled orders "
                 f"of customer {cid}?")
            a = money(sum(db.order_total_cents(o) for o in ids))
        tasks.append(Task(f"T{t:04d}", q, a, kind))
    return tasks


def check(answer: str | None, gold: str) -> bool:
    if answer is None:
        return False
    a = answer.strip().replace("$", "").replace(",", "")
    try:
        return abs(float(a) - float(gold)) < 0.005
    except ValueError:
        return a.lower() == gold.lower()
