"""
ERPNext Service für LiveKit Agent (async, analog EmailService).

Bietet:
  - search_customer / get_customer / create_customer
  - find_item
  - create_quotation
  - create_invoice_draft / submit_and_send_invoice
  - get_open_invoices
"""
import os
import json
import logging
from typing import Optional
import httpx

logger = logging.getLogger("priv-agent")

ERPNEXT_URL = os.getenv("ERPNEXT_URL", "http://erpnext.fastlane-ai.ch:8080").rstrip("/")
ERPNEXT_API_KEY = os.getenv("ERPNEXT_API_KEY", "")
ERPNEXT_API_SECRET = os.getenv("ERPNEXT_API_SECRET", "")
ERPNEXT_COMPANY = os.getenv("ERPNEXT_COMPANY", "Fastlane AI GmbH")
ERPNEXT_CURRENCY = os.getenv("ERPNEXT_DEFAULT_CURRENCY", "CHF")
ERPNEXT_TAX_TEMPLATE = os.getenv("ERPNEXT_DEFAULT_TAX_TEMPLATE", "CH MwSt 8.1% - FAG")
ERPNEXT_INVOICE_THRESHOLD = float(os.getenv("ERPNEXT_INVOICE_AMOUNT_CONFIRM_THRESHOLD", "5000"))
ERPNEXT_TIMEOUT = float(os.getenv("ERPNEXT_TIMEOUT", "20"))


class ERPNextService:
    """Async REST API Client für ERPNext v16."""

    def __init__(self):
        self.base_url = ERPNEXT_URL
        self.company = ERPNEXT_COMPANY
        self.currency = ERPNEXT_CURRENCY
        self.tax_template = ERPNEXT_TAX_TEMPLATE
        self.invoice_threshold = ERPNEXT_INVOICE_THRESHOLD
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"🔗 ERPNext Service: {self.base_url} / {self.company}")

    async def start(self):
        if not (ERPNEXT_API_KEY and ERPNEXT_API_SECRET):
            logger.warning("⚠️  ERPNEXT_API_KEY/SECRET nicht gesetzt")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"token {ERPNEXT_API_KEY}:{ERPNEXT_API_SECRET}",
                "Content-Type": "application/json",
            },
            timeout=ERPNEXT_TIMEOUT,
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs) -> tuple[bool, dict | str]:
        if not self._client:
            await self.start()
        try:
            r = await self._client.request(method, path, **kwargs)
            if r.status_code >= 400:
                logger.error(f"❌ ERPNext {method} {path} → {r.status_code}: {r.text[:300]}")
                if r.status_code in (401, 403):
                    return False, "Zugriff auf ERPNext ist nicht möglich."
                return False, f"ERPNext meldete einen Fehler ({r.status_code})."
            return True, r.json()
        except httpx.TimeoutException:
            logger.error(f"❌ ERPNext Timeout: {method} {path}")
            return False, "ERPNext antwortet im Moment nicht."
        except Exception as e:
            logger.error(f"❌ ERPNext Fehler: {e}")
            return False, "Ein Fehler ist bei ERPNext aufgetreten."

    # ========================================================================
    # CUSTOMER
    # ========================================================================

    async def search_customer(self, query: str) -> tuple[bool, list[dict] | str]:
        ok, data = await self._request("GET", "/api/resource/Customer", params={
            "filters": json.dumps([["customer_name", "like", f"%{query}%"]]),
            "fields": json.dumps(["name", "customer_name", "customer_group"]),
            "limit_page_length": 10,
        })
        if not ok:
            return False, data
        return True, data.get("data", [])

    async def get_customer(self, name: str) -> tuple[bool, dict | str]:
        ok, data = await self._request("GET", f"/api/resource/Customer/{name}")
        if not ok:
            return False, data
        cust = data.get("data", {})
        details = {
            "name": cust.get("name"),
            "customer_name": cust.get("customer_name"),
            "customer_group": cust.get("customer_group"),
            "email": None,
            "phone": None,
            "address": None,
        }
        if cust.get("customer_primary_contact"):
            ok2, contact = await self._request("GET", f"/api/resource/Contact/{cust['customer_primary_contact']}")
            if ok2:
                c = contact.get("data", {})
                if c.get("email_ids"):
                    details["email"] = c["email_ids"][0].get("email_id")
                if c.get("phone_nos"):
                    details["phone"] = c["phone_nos"][0].get("phone")
        if cust.get("customer_primary_address"):
            ok3, addr = await self._request("GET", f"/api/resource/Address/{cust['customer_primary_address']}")
            if ok3:
                a = addr.get("data", {})
                details["address"] = f"{a.get('address_line1', '')}, {a.get('pincode', '')} {a.get('city', '')}".strip(", ")
        return True, details

    async def create_customer(
        self,
        customer_name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        customer_group: str = "Commercial",
    ) -> tuple[bool, str]:
        # Duplikat-Check
        ok, results = await self.search_customer(customer_name)
        if ok and isinstance(results, list):
            for r in results:
                if r.get("customer_name", "").lower() == customer_name.lower():
                    return False, f"Kunde existiert bereits: {r['name']}"

        ok, data = await self._request("POST", "/api/resource/Customer", json={
            "customer_name": customer_name,
            "customer_type": "Company",
            "customer_group": customer_group,
            "territory": "Switzerland",
        })
        if not ok:
            return False, data
        customer_id = data["data"]["name"]
        logger.info(f"✅ Customer angelegt: {customer_id}")

        if email or phone:
            payload = {
                "first_name": customer_name[:140],
                "links": [{"link_doctype": "Customer", "link_name": customer_id}],
            }
            if email:
                payload["email_ids"] = [{"email_id": email, "is_primary": 1}]
            if phone:
                payload["phone_nos"] = [{"phone": phone, "is_primary_phone": 1}]

            ok2, contact_data = await self._request("POST", "/api/resource/Contact", json=payload)
            if ok2:
                contact_id = contact_data["data"]["name"]
                await self._request("PUT", f"/api/resource/Customer/{customer_id}",
                                    json={"customer_primary_contact": contact_id})
                logger.info(f"   Contact verknüpft: {contact_id}")
        return True, customer_id

    # ========================================================================
    # ITEM
    # ========================================================================

    async def find_item(self, query: str) -> tuple[bool, list[dict] | str]:
        ok, data = await self._request("GET", "/api/resource/Item", params={
            "filters": json.dumps([
                ["item_name", "like", f"%{query}%"],
                ["disabled", "=", 0],
            ]),
            "fields": json.dumps(["item_code", "item_name", "stock_uom", "standard_rate"]),
            "limit_page_length": 10,
        })
        if not ok:
            return False, data
        return True, data.get("data", [])

    # ========================================================================
    # QUOTATION
    # ========================================================================

    async def create_quotation(self, customer_id: str, items: list[dict]) -> tuple[bool, dict | str]:
        ok, data = await self._request("POST", "/api/resource/Quotation", json={
            "party_name": customer_id,
            "quotation_to": "Customer",
            "company": self.company,
            "currency": self.currency,
            "items": items,
            "taxes_and_charges": self.tax_template,
        })
        if not ok:
            return False, data
        q = data["data"]
        return True, {
            "name": q["name"],
            "grand_total": q.get("grand_total", 0),
            "currency": q.get("currency", self.currency),
        }

    # ========================================================================
    # SALES INVOICE
    # ========================================================================

    async def create_invoice_draft(self, customer_id: str, items: list[dict]) -> tuple[bool, dict | str]:
        ok, data = await self._request("POST", "/api/resource/Sales Invoice", json={
            "customer": customer_id,
            "company": self.company,
            "currency": self.currency,
            "items": items,
            "taxes_and_charges": self.tax_template,
        })
        if not ok:
            return False, data
        inv = data["data"]
        return True, {
            "name": inv["name"],
            "grand_total": inv.get("grand_total", 0),
            "currency": inv.get("currency", self.currency),
            "items_count": len(inv.get("items", [])),
        }

    async def submit_invoice(self, invoice_name: str) -> tuple[bool, str]:
        ok, data = await self._request("GET", f"/api/resource/Sales Invoice/{invoice_name}")
        if not ok:
            return False, data
        doc = data.get("data", {})
        ok2, result = await self._request("POST", "/api/method/frappe.client.submit",
                                          json={"doc": json.dumps(doc)})
        if not ok2:
            return False, result
        logger.info(f"✅ Invoice submitted: {invoice_name}")
        return True, invoice_name

    async def send_invoice_email(self, invoice_name: str, recipient: str) -> tuple[bool, str]:
        ok, data = await self._request(
            "POST",
            "/api/method/frappe.core.doctype.communication.email.email.make",
            json={
                "doctype": "Sales Invoice",
                "name": invoice_name,
                "subject": f"Rechnung {invoice_name} - {self.company}",
                "content": (
                    f"<p>Sehr geehrte Damen und Herren,</p>"
                    f"<p>im Anhang erhalten Sie unsere Rechnung <b>{invoice_name}</b>.</p>"
                    f"<p>Zahlbar innert 30 Tagen.</p>"
                    f"<p>Freundliche Grüsse<br>{self.company}</p>"
                ),
                "recipients": recipient,
                "send_email": 1,
                "print_format": "Standard",
                "attach_document_print": 1,
                "communication_medium": "Email",
            },
        )
        if not ok:
            return False, data
        logger.info(f"✅ Invoice email an {recipient}: {invoice_name}")
        return True, recipient

    # ========================================================================
    # READ-ONLY HELPER
    # ========================================================================

    async def get_open_invoices(self, customer_id: str) -> tuple[bool, list[dict] | str]:
        ok, data = await self._request("GET", "/api/resource/Sales Invoice", params={
            "filters": json.dumps([
                ["customer", "=", customer_id],
                ["status", "in", ["Unpaid", "Overdue", "Partly Paid"]],
                ["docstatus", "=", 1],
            ]),
            "fields": json.dumps(["name", "grand_total", "outstanding_amount", "due_date", "status"]),
            "limit_page_length": 20,
        })
        if not ok:
            return False, data
        return True, data.get("data", [])
