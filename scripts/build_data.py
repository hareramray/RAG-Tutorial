"""Rebuild the fictional, shareable teaching corpus. No company data is used."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    ("benefits_leave", "Leave and learning benefits", "people", "internal", """
Northstar Analytics is a fictional company used only for this RAG workshop.
Full-time employees receive 24 days of paid annual leave per calendar year. Leave accrues at two days per completed month. Part-time annual leave is prorated. Employees request planned leave in the People Portal at least five working days in advance. A manager must approve the request before travel is booked. Sick leave is recorded separately and does not reduce annual leave.
Employees may carry forward up to 5 unused annual leave days into the following year. Carried days expire on March 31 and are used before the current year's allowance. Any remaining carried balance after March 31 expires; it is not paid out. The HR team handles corrections to leave balances.
Each employee has an annual learning budget of INR 30000 for approved courses, books, and conferences. Obtain manager approval before purchase and submit an itemized receipt within 30 days. The learning budget does not carry forward. This policy does not describe parental leave, contractor leave, or country-specific statutory benefits; contact HR for those questions.
"""),
    ("remote_work", "Remote work policy", "people", "internal", """
Northstar Analytics allows eligible employees to work remotely up to 3 days per week with manager approval. Teams coordinate two office collaboration days each week. Remote days are not additional leave: employees remain available during their agreed working hours. New employees agree their schedule with their manager during onboarding.
Core collaboration hours are 11:00 to 16:00 India Standard Time on working days. Employees may arrange the remainder of their working day with their manager. Changes to a remote schedule should be recorded in the team calendar before the week begins. An exception for more than three remote days requires approval from both the manager and the People team.
Use a company-managed device and the company VPN when accessing internal services outside the office. Public Wi-Fi is permitted only through the VPN. Work from another country requires advance approval from the People and Security teams. This document does not authorize permanent international relocation or describe home-office equipment reimbursement.
"""),
    ("travel_expenses", "Business travel and expenses", "finance", "internal", """
Northstar Analytics reimburses pre-approved business travel expenses. Employees must obtain their manager's approval and a trip reference before booking. Domestic flights should be economy class. Hotel reimbursement is capped at INR 6000 per night for domestic travel, excluding taxes. Meals during domestic business travel are capped at INR 1500 per day.
Submit expense claims through the Finance Portal within 30 days after the trip ends. Every claim needs a receipt, the trip reference, the business purpose, and the approval record. Finance reviews complete claims within 10 working days. Missing receipts require a written explanation and finance approval; reimbursement is not automatic.
Personal holidays attached to business trips are paid for by the employee. Currency conversion uses the transaction rate on the card statement. This policy covers business travel only. It does not set the annual learning allowance or home-office benefits. Questions about training reimbursement should be checked against the learning benefits policy, not inferred from meal or hotel limits.
"""),
    ("security_access", "Access and account security", "engineering", "internal", """
All Northstar Analytics employee accounts require multi-factor authentication, also called MFA. Access to internal systems uses single sign-on. Employees must use the company password manager and must never paste API secrets into notebooks, tickets, or chat messages. Local development secrets belong in a .env file excluded from version control.
Request access through the Service Desk with a business justification and the system owner's approval. Access is granted with the smallest necessary scope and reviewed every 90 days. Elevated production access is time limited. The system owner revokes access when the business need ends; employee offboarding revokes accounts on the final working day.
If an API key is exposed, revoke the key, issue a replacement, and notify Security immediately. Removing the key from a file is insufficient because copies may remain in history. Never share a production token for a training demonstration. A forgotten password follows the Service Desk reset process. This document does not contain real credentials, and no training exercise requires reading an employee's private files.
"""),
    ("product_plans", "Northstar Cloud subscription plans", "product", "public", """
Northstar Cloud is a fictional analytics product. Its Team plan costs INR 2400 per user per month when billed annually. A Team subscription requires a minimum of 5 users. The advertised monthly figure is a per-user equivalent, not a monthly billing option. Taxes are additional. The Team plan includes shared dashboards, scheduled reports, and standard email support.
The Enterprise plan uses custom annual pricing negotiated with Sales. It adds single sign-on, audit log export, dedicated onboarding, and priority support. Neither document nor training assistant should invent an Enterprise price. Contact Sales for a quote based on seat count and requirements.
Default event-data retention is 90 days for Team and 365 days for Enterprise. The data retention policy describes deletion and backup behavior in more detail. Plan limits are enforced per workspace, not per individual user. API request rates are documented in the API limits guide. Upgrading a plan does not restore data that has already expired. Discounts, promotional codes, and individual contracts are outside this public guide.
"""),
    ("refunds", "Refund and cancellation policy", "billing", "public", """
For the fictional Northstar Cloud service, a refund may be requested within 14 days of purchasing an annual subscription, provided the subscription is unused. Both conditions must be satisfied: the purchase must be inside the 14-day window and the subscription must have no usage. A used subscription is not eligible under this policy even if it is less than 14 days old.
Send the invoice number and workspace identifier to Billing through the support portal. Billing reviews an eligible request within 5 working days. An approved refund is returned to the original payment method. Bank processing times may vary. Do not promise immediate receipt of funds.
Customers can cancel renewal at any time from workspace billing settings. Cancellation stops the next renewal; current service continues until the paid term ends. Cancellation does not itself trigger a refund. This standard policy does not cover individually negotiated Enterprise contracts, which must be reviewed by Billing. There is no documented 30-day unconditional money-back guarantee.
"""),
    ("api_limits", "API limits and retry behavior", "engineering", "public", """
The Northstar Cloud API enforces request limits per workspace. The Team plan allows 600 requests per minute. The Enterprise plan allows 3000 requests per minute. These are request counts, not token limits. All clients sharing one workspace contribute to the same limit. A burst of requests can therefore affect another client in that workspace.
When the request limit is exceeded, the API returns HTTP 429. Clients should respect the Retry-After header and retry with exponential backoff and random jitter. Repeated immediate retries will not remove the limit. Limit retries and surface a useful error to the caller when the retry budget is exhausted. Do not retry invalid authentication responses as though they were rate limits.
Bulk export jobs are limited to 2 concurrent jobs per workspace on either plan. Paginated list endpoints accept a maximum page size of 100 records. For sustained higher request rates, contact Support to discuss capacity. This guide contains fictional Northstar API behavior, not the rate limits of Anthropic's API, which depend on the actual account and model.
"""),
    ("incident_response", "Internal incident response playbook", "engineering", "restricted", """
This fictional internal playbook covers Northstar service incidents. A severity-one incident is an outage or a critical security event affecting production customers. The on-call engineer must acknowledge a severity-one alert within 15 minutes. The incident commander coordinates investigation, mitigation, and communication. Acknowledgement is not a promise to fully resolve the issue within 15 minutes.
During an active severity-one incident, post customer-facing status updates every 30 minutes. The engineering lead chooses an incident commander and a separate communications owner. Preserve relevant operational evidence while avoiding disclosure of customer secrets. Escalate suspected credential exposure to Security immediately.
After recovery, complete a blameless incident review within 3 working days. Record the timeline, contributing factors, customer impact, and follow-up owners. Restricted incident details are visible only to authorized engineering responders. The public support service-level guide describes customer support response targets; those targets differ from the internal on-call acknowledgement target. Never infer a recovery-time guarantee from an acknowledgement deadline.
"""),
    ("data_retention", "Data retention and deletion", "product", "public", """
Northstar Cloud retains event data for 90 days on the Team plan and 365 days on the Enterprise plan. Retention is measured from event ingestion time. Events older than the workspace's retention window are removed from the active query store. Upgrading from Team to Enterprise extends retention for data still present; it cannot recover already expired events.
Workspace administrators can request deletion of a workspace through the support portal. After identity and authority checks, active workspace data is deleted within 7 days of an approved request. Encrypted backup copies expire within 30 days. Backup expiry and deletion from the active store are different operations and should not be described as immediate removal from every copy.
Customers should export required reports before cancellation or deletion. A billing cancellation stops renewal but does not create an immediate deletion request. Individually negotiated retention terms must be confirmed with the account owner. The public standard policy does not specify legal-hold exceptions or a service in a particular country. Avoid guessing about those matters.
"""),
    ("onboarding", "New employee onboarding", "people", "internal", """
On day one at Northstar Analytics, a new employee meets their manager, completes identity verification, and activates their single sign-on account with MFA. IT provides a company-managed laptop and access to the password manager. Credentials must never be placed in shared training notebooks. Access requests still require a system owner even during onboarding.
During the first week, the employee completes security awareness training, reads the remote work policy, and agrees on two office collaboration days with their manager. A buddy helps the employee find internal documentation. Remote work may be used up to three days each week with manager approval; onboarding itself does not grant extra remote days.
By day 30, the manager and employee review role goals and training needs. The annual learning budget is described in the benefits policy and requires approval before purchase. This onboarding checklist links related policies but does not replace their detailed conditions. Contractor onboarding and international employment arrangements follow separate processes outside this workshop corpus.
"""),
    ("release_notes", "Northstar Cloud release 2.4", "product", "public", """
Northstar Cloud release 2.4 became generally available on September 15, 2026. The release introduced scheduled CSV exports for Team and Enterprise workspaces. Workspace administrators can schedule a daily or weekly export to an approved destination. Each workspace may configure up to 10 active export schedules.
The release also added a dashboard freshness indicator that displays the time of the most recent successful data refresh. A failed refresh leaves the previous successful timestamp visible and shows a warning. The indicator is not a guarantee of real-time ingestion.
Version 2.4 did not change the documented Team or Enterprise request-per-minute limits. Bulk exports remain subject to the API concurrency limit. Previously created dashboards continue to work. Features planned for future releases are not commitments and are not listed here. For questions about a future version or an announced date outside this release, the training assistant should acknowledge that the knowledge base lacks evidence.
"""),
    ("support_sla", "Customer support response targets", "support", "public", """
Northstar Cloud Team customers receive email support during working hours, Monday through Friday, 09:00 to 18:00 India Standard Time, excluding published holidays. The Team target for a first response to a standard ticket is one business day. A first response means acknowledgement and initial triage, not full resolution.
Enterprise customers receive priority support. For a critical production outage, the Enterprise first-response target is 1 hour, with coverage 24 hours a day, seven days a week. Standard noncritical Enterprise tickets have a first-response target of 4 business hours. A customer's individual contract may override these standard targets.
Include the workspace identifier, impact summary, timestamp, and a sanitized error description in a support request. Never include API keys or raw personal customer data. During a published service incident, check the status page for progress. The internal engineering incident acknowledgement target is separate from these customer support targets. The guide does not promise a universal time to resolution or automatic compensation.
"""),
]

docs = [{"id": key, "text": " ".join(text.split()), "metadata": {
    "title": title, "source": key + ".md", "department": dept,
    "visibility": visibility, "effective_date": "2026-10-01",
}} for key, title, dept, visibility, text in SOURCES]

EVAL = [
    ("dev_leave", "How many annual paid leave days do full-time employees receive?", ["benefits_leave"], "24 days per calendar year.", "dev"),
    ("dev_carry", "How many unused leave days can carry forward, and when do they expire?", ["benefits_leave"], "Up to 5 days, expiring on March 31.", "dev"),
    ("dev_remote", "How many days a week may I work remotely and who approves?", ["remote_work"], "Up to 3 days per week with manager approval.", "dev"),
    ("dev_retry", "What should a client do after HTTP 429?", ["api_limits"], "Respect Retry-After and use bounded exponential backoff with jitter.", "dev"),
    ("dev_refund", "Can I get a refund for an unused annual subscription bought ten days ago?", ["refunds"], "It meets the unused and within-14-days conditions; submit to Billing for review.", "dev"),
    ("dev_compare", "Compare Team and Enterprise event retention and API request limits.", ["data_retention", "api_limits"], "Team 90 days and 600 requests/minute; Enterprise 365 days and 3000 requests/minute.", "dev"),
    ("dev_unknown", "What is Northstar's parental leave allowance?", [], "I do not have enough evidence to answer.", "dev"),
    ("test_learning", "How much can an employee spend each year on approved learning?", ["benefits_leave"], "INR 30000 per year with manager approval before purchase.", "test"),
    ("test_price", "What does the Team subscription cost and what is the minimum seat count?", ["product_plans"], "INR 2400 per user per month billed annually, minimum 5 users, taxes additional.", "test"),
    ("test_delete", "After approved workspace deletion, when does active data disappear and when do backups expire?", ["data_retention"], "Active data within 7 days; encrypted backup copies within 30 days.", "test"),
    ("test_incident", "What are the acknowledgement and update intervals for an internal severity-one incident?", ["incident_response"], "Acknowledge within 15 minutes and customer-facing updates every 30 minutes.", "test"),
    ("test_support", "What is the Enterprise first-response target for a critical outage?", ["support_sla"], "1 hour, with 24/7 coverage; this is not a resolution guarantee.", "test"),
    ("test_refund_used", "Is a used annual subscription refundable five days after purchase under the standard policy?", ["refunds"], "No. Both within 14 days and unused are required.", "test"),
    ("test_release", "How many scheduled exports can a workspace configure in release 2.4?", ["release_notes"], "Up to 10 active export schedules.", "test"),
    ("test_unknown", "What is the CEO's private mobile number?", [], "I do not have enough evidence to answer.", "test"),
]
questions = [{"id": key, "question": q, "relevant_ids": ids,
              "reference_answer": answer, "split": split, "answerable": bool(ids)}
             for key, q, ids, answer, split in EVAL]
for filename, payload in [("knowledge_base.json", docs), ("eval_questions.json", questions)]:
    (ROOT / "data" / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Created {len(docs)} fictional documents and {len(questions)} evaluation questions.")
