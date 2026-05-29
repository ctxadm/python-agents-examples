"""
ERPNext Service für LiveKit Agent (async, analog EmailService).

Bietet:
  - search_customer / get_customer / create_customer / set_customer_contact
  - get_customer_email_from_invoice
  - find_item
  - create_quotation
  - create_invoice_draft / submit_invoice / send_invoice_email
  - get_open_invoices
"""
import os
import re
import json
import difflib
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

# Firmensuffixe für tolerantes Customer-Name-Matching
_COMPANY_SUFFIXES = (
    " ag", " gmbh", " sa", " sarl", " inc", " ltd", " llc",
    " gbr", " kg", " ohg", " e.k.", " e.v.", " ug",
)

# Mindest-Ähnlichkeits-Score, ab dem ein Fuzzy-Treffer als "exakt" gilt
_HIGH_CONFIDENCE_THRESHOLD = 0.85

# Minimum-Ähnlichkeit, ab der ein Customer überhaupt in die Fuzzy-Liste kommt
_MIN_FUZZY_SCORE = 0.4


def _normalize_customer_name(name: str) -> str:
    """Lowercase + gängigen Firmensuffix abschneiden für tolerantes Matching."""
    s = (name or "").lower().strip()
    for suf in _COMPANY_SUFFIXES:
        if s.endswith(suf):
            s = s[:-len(suf)].strip()
            break
    return s


def _normalize_phone_e164(phone: str, default_country_code: str = "+41") -> str:
    """
    Normalisiert eine Telefonnummer zu E.164-Format (CH-Default: +41).
    Beispiele:
      "044 123 45 67"     -> "+41441234567"
      "+41 44 123 45 67"  -> "+41441234567"
      "0041 44 123 45 67" -> "+41441234567"
      "+41441234567"      -> "+41441234567"
    """
    if not phone:
        return ""
    cleaned = re.sub(r"[^\d+]", "", phone.strip())
    if not cleaned:
        return ""
    if cleaned.startswith("+"):
        return cleaned
    if cleaned.startswith("00"):
        return "+" + cleaned[2:]
    if cleaned.startswith("0"):
        return default_country_code + cleaned[1:]
    return default_country_code + cleaned


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
                if r.status_code == 404:
                    return False, "Das angeforderte Dokument existiert nicht in ERPNext."
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

    async def search_customer(self, query: str) -> tuple[bool, dict | str]:
        """
        Suche Customers mit mehrstufiger Strategie:
          1. Exact (case-insensitive + Firmensuffix-tolerant)
          2. Substring-Präfix-Suche (4-Zeichen-Tokens, schnell)
          2b. Fetch-All Fallback wenn Stufe 2 leer (typo-tolerant an jeder Position)
          3. Re-Ranking via difflib.SequenceMatcher;
             Top-1 mit Score >= 0.85 wird als exact_match gewertet.

        Returns:
            (True, {"exact_match": bool, "results": list[dict]})
        """
        query_clean = (query or "").strip()
        if not query_clean:
            return True, {"exact_match": False, "results": []}
        query_norm = _normalize_customer_name(query_clean)

        # === Stufe 1: Case-insensitive Substring + Suffix-tolerantes Match ===
        ok, data = await self._request("GET", "/api/resource/Customer", params={
            "filters": json.dumps([["customer_name", "like", f"%{query_clean}%"]]),
            "fields": json.dumps(["name", "customer_name", "customer_group"]),
            "limit_page_length": 20,
        })
        if not ok:
            return False, data

        candidates = data.get("data", [])
        exact_matches = [
            c for c in candidates
            if _normalize_customer_name(c["customer_name"]) == query_norm
        ]
        if len(exact_matches) == 1:
            logger.info(f"   🎯 Exact-Match (case/suffix-tolerant): {exact_matches[0]['customer_name']}")
            return True, {"exact_match": True, "results": exact_matches}

        # === Stufe 2: Substring-Präfix-Suche (typo-tolerant am Wortende) ===
        tokens = [t for t in re.split(r"[\s\-_.]+", query_clean) if len(t) >= 3]
        if not tokens:
            tokens = [query_clean]
        short_tokens = [t[:4] if len(t) > 4 else t for t in tokens]
        or_filters = [["customer_name", "like", f"%{t}%"] for t in short_tokens]

        ok, data = await self._request("GET", "/api/resource/Customer", params={
            "or_filters": json.dumps(or_filters),
            "fields": json.dumps(["name", "customer_name", "customer_group"]),
            "limit_page_length": 50,
        })
        if not ok:
            return False, data
        raw_results = data.get("data", [])

        # === Stufe 2b: Fetch-All Fallback (typo-tolerant an JEDER Position) ===
        if not raw_results:
            logger.info(f"   ⚠️ Stufe 2 leer für '{query_clean}' – Fallback Fetch-All")
            ok, data = await self._request("GET", "/api/resource/Customer", params={
                "fields": json.dumps(["name", "customer_name", "customer_group"]),
                "limit_page_length": 0,
            })
            if not ok:
                return False, data
            raw_results = data.get("data", [])
            logger.info(f"   📥 Fetch-All: {len(raw_results)} Customers geladen")

        if not raw_results:
            logger.info(f"   ❌ Kein Treffer für '{query_clean}'")
            return True, {"exact_match": False, "results": []}

        # === Stufe 3: Re-Ranking via SequenceMatcher ===
        scored = [
            (
                difflib.SequenceMatcher(
                    None,
                    query_norm,
                    _normalize_customer_name(c["customer_name"]),
                ).ratio(),
                c,
            )
            for c in raw_results
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_customer = scored[0]
        logger.info(
            f"   🔍 Top-Score: {top_customer['customer_name']} = {top_score:.2f} "
            f"(Query: '{query_clean}')"
        )

        if top_score < _MIN_FUZZY_SCORE:
            logger.info(f"   ❌ Top-Score {top_score:.2f} unter Schwelle {_MIN_FUZZY_SCORE}")
            return True, {"exact_match": False, "results": []}

        if top_score >= _HIGH_CONFIDENCE_THRESHOLD:
            logger.info(f"   🎯 High-Confidence-Match: {top_customer['customer_name']}")
            return True, {"exact_match": True, "results": [top_customer]}

        return True, {
            "exact_match": False,
            "results": [c for s, c in scored[:5] if s >= _MIN_FUZZY_SCORE],
        }

    async def get_customer(self, name: str) -> tuple[bool, dict | str]:
        """
        Liest Customer-Details inkl. Email/Phone mit 3-stufigem Fallback.
        Liefert zusätzlich address_line1, pincode und city einzeln,
        damit die Adresse im Tool-Layer Voice-/Chat-konform formatiert werden kann.
        """
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
            "address": None,         # combined (kompatibel)
            "address_line1": None,   # neu: Strasse separat
            "pincode": None,         # neu: PLZ separat
            "city": None,            # neu: Ort separat
        }

        # --- Stufe 1: Direktes email_id-Feld am Customer ---
        if cust.get("email_id"):
            details["email"] = cust["email_id"]
            logger.info(f"   📧 Email aus Customer.email_id: {details['email']}")

        # --- Stufe 2: customer_primary_contact (klassischer Weg) ---
        if cust.get("customer_primary_contact"):
            ok2, contact = await self._request(
                "GET", f"/api/resource/Contact/{cust['customer_primary_contact']}"
            )
            if ok2:
                c = contact.get("data", {})
                if not details["email"] and c.get("email_ids"):
                    details["email"] = c["email_ids"][0].get("email_id")
                    logger.info(f"   📧 Email aus Primary Contact: {details['email']}")
                if c.get("phone_nos"):
                    details["phone"] = c["phone_nos"][0].get("phone")

        # --- Stufe 3: Fallback – Contacts via Dynamic Link suchen ---
        if not details["email"] or not details["phone"]:
            ok3, contacts_data = await self._request(
                "GET", "/api/resource/Contact",
                params={
                    "filters": json.dumps([
                        ["Dynamic Link", "link_doctype", "=", "Customer"],
                        ["Dynamic Link", "link_name", "=", cust.get("name")],
                    ]),
                    "fields": json.dumps(["name", "email_id", "phone"]),
                    "limit_page_length": 10,
                },
            )
            if ok3:
                for contact in contacts_data.get("data", []):
                    if not details["email"] and contact.get("email_id"):
                        details["email"] = contact["email_id"]
                        logger.info(
                            f"   📧 Email via Dynamic-Link-Fallback aus '{contact['name']}': {details['email']}"
                        )
                    if not details["phone"] and contact.get("phone"):
                        details["phone"] = contact["phone"]
                    if details["email"] and details["phone"]:
                        break

        # --- Adresse (jetzt mit Einzelfeldern) ---
        if cust.get("customer_primary_address"):
            ok4, addr = await self._request(
                "GET", f"/api/resource/Address/{cust['customer_primary_address']}"
            )
            if ok4:
                a = addr.get("data", {})
                details["address_line1"] = a.get("address_line1") or None
                details["pincode"] = a.get("pincode") or None
                details["city"] = a.get("city") or None
                # Kombiniertes Feld für Backwards-Compat
                line1 = a.get("address_line1", "")
                pin = a.get("pincode", "")
                city = a.get("city", "")
                combined = f"{line1}, {pin} {city}".strip(", ").strip()
                details["address"] = combined or None

        return True, details

    async def get_customer_email_from_invoice(self, invoice_name: str) -> Optional[str]:
        """Holt die primäre Email des Customers, der zu einer Invoice gehört."""
        ok, data = await self._request("GET", f"/api/resource/Sales Invoice/{invoice_name}")
        if not ok:
            return None
        customer_id = data.get("data", {}).get("customer") if isinstance(data, dict) else None
        if not customer_id:
            return None
        ok2, details = await self.get_customer(customer_id)
        if not ok2 or not isinstance(details, dict):
            return None
        return details.get("email")

    async def create_customer(
        self,
        customer_name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        customer_group: str = "Commercial",
    ) -> tuple[bool, str]:
        ok, results = await self.search_customer(customer_name)
        if ok and isinstance(results, dict) and results.get("exact_match"):
            existing = results["results"][0]
            return False, f"Kunde existiert bereits: {existing['name']}"

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
                phone_e164 = _normalize_phone_e164(phone)
                payload["phone_nos"] = [{
                    "phone": phone_e164 or phone,
                    "is_primary_phone": 1,
                    "is_primary_mobile_no": 1,
                }]

            ok2, contact_data = await self._request("POST", "/api/resource/Contact", json=payload)
            if ok2:
                contact_id = contact_data["data"]["name"]
                await self._request("PUT", f"/api/resource/Customer/{customer_id}",
                                    json={"customer_primary_contact": contact_id})
                logger.info(f"   Contact verknüpft: {contact_id}")
        return True, customer_id

    async def set_customer_contact(
        self,
        customer_id: str,
        phone: str,
        email: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Setzt den primären Kontakt (Telefon + optional Email) eines bestehenden Kunden.

        Phase 1: Funktioniert NUR wenn der Kunde aktuell KEINEN primären Kontakt hat
        (typischer Fall nach CSV-Import). Wenn customer_primary_contact bereits gesetzt
        ist, wird abgebrochen und ein klarer Hinweis zurückgegeben.

        Telefon wird automatisch zu E.164 normalisiert (CH-Default: +41).

        Args:
            customer_id: Customer-ID aus ERPNext (exakter Name)
            phone: Telefonnummer (beliebiges Format, wird normalisiert)
            email: E-Mail-Adresse (optional)

        Returns:
            (True, contact_id) bei Erfolg
            (False, fehlermeldung) bei Fehler oder bereits vorhandenem Primary Contact
        """
        # 1) Customer lesen → primary_contact prüfen
        ok, data = await self._request("GET", f"/api/resource/Customer/{customer_id}")
        if not ok:
            return False, data if isinstance(data, str) else "Kunde nicht gefunden."

        cust = data.get("data", {})
        if cust.get("customer_primary_contact"):
            existing = cust["customer_primary_contact"]
            logger.warning(f"   ⚠️ Customer hat bereits Primary Contact: {existing}")
            return False, (
                f"Der Kunde hat bereits einen primären Kontakt ({existing}). "
                f"Das Aktualisieren bestehender Kontakte ist aktuell nicht freigegeben."
            )

        customer_name = cust.get("customer_name") or customer_id

        # 2) Telefon normalisieren
        phone_e164 = _normalize_phone_e164(phone)
        if not phone_e164:
            return False, "Die Telefonnummer ist ungültig."

        # 3) Neuen Contact anlegen
        payload = {
            "first_name": customer_name[:140],
            "links": [{"link_doctype": "Customer", "link_name": customer_id}],
            "phone_nos": [{
                "phone": phone_e164,
                "is_primary_phone": 1,
                "is_primary_mobile_no": 1,
            }],
        }
        if email:
            payload["email_ids"] = [{"email_id": email, "is_primary": 1}]

        ok, contact_data = await self._request("POST", "/api/resource/Contact", json=payload)
        if not ok:
            return False, contact_data
        contact_id = contact_data["data"]["name"]
        logger.info(f"✅ Contact angelegt: {contact_id} für {customer_id}")

        # 4) Customer verlinken
        ok, link_data = await self._request(
            "PUT",
            f"/api/resource/Customer/{customer_id}",
            json={"customer_primary_contact": contact_id},
        )
        if not ok:
            return False, (
                f"Kontakt {contact_id} angelegt, aber Verknüpfung mit Kunde fehlgeschlagen: {link_data}"
            )

        logger.info(f"   ✅ Customer {customer_id} → Primary Contact {contact_id}")
        return True, contact_id

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
            if isinstance(data, str) and "existiert nicht" in data:
                return False, (
                    f"Rechnung '{invoice_name}' existiert nicht in ERPNext. "
                    f"Wahrscheinlich wurde kein Rechnungsentwurf erstellt. "
                    f"Rufe zuerst erp_create_invoice_draft auf und verwende den dort "
                    f"zurückgegebenen Namen."
                )
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
            "/api/method/frappe.core.doctype.communication.email.make",
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
