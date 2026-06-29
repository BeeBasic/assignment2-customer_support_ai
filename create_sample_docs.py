"""
create_sample_docs.py — Generate sample PDF company documents for RAG
=======================================================================

PURPOSE
-------
Creates the four company documents used by the RAG system using fpdf2
with the modern API (fpdf2 >= 2.5.2).

HOW TO RUN
----------
From customer_support_ai directory:
    python create_sample_docs.py

OUTPUT
------
Creates four PDF files in the documents/ folder:
  - company_policy.pdf
  - pricing_guide.pdf
  - technical_manual.pdf
  - faq.pdf
"""

import os
import sys

# ── Install fpdf2 if not present ─────────────────────────────────────────────
try:
    from fpdf import FPDF, XPos, YPos
except ImportError:
    print("Installing fpdf2...")
    os.system(f"{sys.executable} -m pip install fpdf2 -q")
    from fpdf import FPDF, XPos, YPos

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
os.makedirs(DOCS_DIR, exist_ok=True)


# ── Document content definitions ─────────────────────────────────────────────

COMPANY_POLICY = """COMPANY POLICY DOCUMENT
TechCorp Customer Support Policy - Version 3.2 - Effective: January 2024

SECTION 1: REFUND POLICY
Customers are eligible for a full refund within 30 days of purchase if the
product does not meet described specifications. Refunds after 30 days are
evaluated on a case-by-case basis and require manager approval. Refunds are
processed within 5-7 business days to the original payment method.

Digital products and activated licenses are non-refundable unless there is
a verifiable technical defect that cannot be resolved by support.

SECTION 2: SUBSCRIPTION CANCELLATION POLICY
Customers may cancel their subscription at any time. Cancellation takes
effect at the end of the current billing cycle. No partial refunds are
issued for unused days in the current billing period.

To cancel a subscription, customers must submit a written request through
the support portal. A confirmation email will be sent within 24 hours.
Cancellation requests require supervisor approval before processing.

SECTION 3: ACCOUNT CLOSURE POLICY
Account closure requests are permanent and irreversible. All customer data
will be deleted within 30 days of closure in accordance with GDPR regulations.
Customers must settle any outstanding balance before an account can be closed.
Account closure requires identity verification and senior support approval
due to its irreversible nature.

SECTION 4: COMPENSATION POLICY
Compensation may be offered in the following situations:
- Service outages lasting more than 4 hours
- Billing errors resulting in overcharges
- Product defects causing significant business impact

Compensation forms: account credits, subscription extensions, or partial refunds.
All compensation offers above $50 require manager approval.

SECTION 5: ESCALATION POLICY
Customers may request escalation to management when:
- Issue is unresolved after two support interactions
- Customer reports significant financial or business impact
- Customer explicitly requests to speak with a manager

Escalations are logged and must be acknowledged within 2 business hours.

SECTION 6: DATA PRIVACY
TechCorp complies with GDPR and CCPA. Customer data is never sold to third
parties. Data is retained for 3 years after account closure for legal purposes.
Customers may request data exports at any time.

SECTION 7: SERVICE LEVEL AGREEMENT (SLA)
Standard support: Response within 24 hours.
Priority support (Pro plan and above): Response within 4 hours.
Enterprise support: Response within 1 hour with dedicated account manager.
"""

PRICING_GUIDE = """PRICING GUIDE
TechCorp Product Pricing - Updated Q1 2024

SUBSCRIPTION PLANS

STARTER PLAN - $9/month (billed monthly) or $90/year (billed annually)
  - Up to 3 users
  - 10 GB storage
  - Email support (24-hour response)
  - Core features: Project management, task tracking, basic reporting
  - Integrations: Slack, Google Drive
  - API access: Not included

PROFESSIONAL PLAN - $29/month (billed monthly) or $290/year (billed annually)
  - Up to 25 users
  - 100 GB storage
  - Priority support (4-hour response)
  - All Starter features PLUS:
    - Advanced analytics and custom dashboards
    - Automated workflows and triggers
    - API access (10,000 calls/month)
    - Integrations: Salesforce, HubSpot, Zapier, Slack, Google Suite
    - Custom branding (white-label reports)

ENTERPRISE PLAN - Custom pricing (contact sales@techcorp.com)
  - Unlimited users and storage
  - Dedicated account manager
  - 1-hour SLA response guarantee
  - All Professional features PLUS:
    - Single Sign-On (SSO / SAML)
    - Advanced security and audit logs
    - Custom integrations and API limits
    - On-premise deployment option
    - Annual business reviews

ADD-ON SERVICES
  Extra Storage: $5/month per 50 GB
  Additional API calls: $10/month per 5,000 calls
  Training sessions: $200 per session (2-hour remote)
  Data migration service: $500 one-time fee
  Custom development: Quoted per project

FREE TRIAL
All plans include a 14-day free trial. No credit card required for Starter
and Professional trials. Enterprise trials require a sales call.

DISCOUNTS
  - Annual billing: Save up to 17% vs monthly
  - Non-profit organizations: 30% discount (verification required)
  - Educational institutions: 50% discount (must be .edu domain)
  - Startup program: First 6 months free for qualifying companies

UPGRADE / DOWNGRADE POLICY
  - Upgrades take effect immediately; prorated difference is charged
  - Downgrades take effect at the end of the current billing cycle

PAYMENT METHODS
Accepted: Visa, Mastercard, American Express, PayPal, Bank Transfer (Enterprise)
All prices shown in USD. International pricing may vary.
"""

TECHNICAL_MANUAL = """TECHNICAL MANUAL
TechCorp Platform - Technical Reference v2.8

CHAPTER 1: SYSTEM REQUIREMENTS
Minimum requirements:
  - OS: Windows 10+, macOS 11+, Ubuntu 20.04+
  - RAM: 4 GB (8 GB recommended)
  - Disk: 2 GB free space
  - Browser: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
  - Internet: 10 Mbps stable connection

Mobile app requirements:
  - iOS 14+ or Android 10+
  - 200 MB free storage

CHAPTER 2: INSTALLATION AND SETUP
Step 1: Create account at app.techcorp.com
Step 2: Verify email address (check spam folder if not received in 5 minutes)
Step 3: Complete onboarding wizard (approximately 10 minutes)
Step 4: Invite team members via Settings > Team > Invite
Step 5: Connect integrations via Settings > Integrations

CHAPTER 3: FILE UPLOAD TROUBLESHOOTING
ISSUE: Application crashes when uploading files
CAUSE: This is typically caused by one of three issues:
  1. File size exceeds the plan limit
     Starter: 25 MB, Pro: 100 MB, Enterprise: 500 MB
  2. Unsupported file format
  3. Browser memory issue with large files

SOLUTION:
  - Check file size against your plan upload limit
  - Supported formats: PDF, DOCX, XLSX, PPTX, PNG, JPG, GIF, MP4, ZIP
  - Clear browser cache and cookies, then retry
  - Try using a different browser
  - For files over 50 MB: use the desktop app instead of browser
  - If issue persists: contact support with browser console logs

CHAPTER 4: API INTEGRATION
Base URL: https://api.techcorp.com/v2
Authentication: Bearer token in Authorization header
Rate limits: Per plan (see Pricing Guide)

Common API errors:
  - 401 Unauthorized: Token expired. Regenerate in Settings > API
  - 429 Too Many Requests: Rate limit exceeded. Use exponential backoff
  - 500 Internal Server Error: Contact support with request ID

CHAPTER 5: PASSWORD RESET
Method 1 (Recommended): Visit app.techcorp.com/reset-password
Method 2: Click Forgot Password on the login page
Method 3 (Account locked): Contact support with photo ID verification

Password requirements: Minimum 8 characters, uppercase, number, and symbol.
Password resets expire after 24 hours.

CHAPTER 6: SINGLE SIGN-ON (SSO)
SSO is available on Enterprise plans only.
Supported protocols: SAML 2.0, OAuth 2.0, OpenID Connect
Contact enterprise@techcorp.com for SSO setup assistance.

CHAPTER 7: DATA EXPORT
Export all data: Settings > Account > Export Data
Formats available: JSON, CSV, XML
Processing time: Up to 24 hours for large accounts
Exports are available for download for 7 days.

CHAPTER 8: KNOWN ISSUES AND STATUS
Current status page: app.techcorp.com/status
Subscribe to status updates: status.techcorp.com/subscribe
Maintenance windows: Sundays 2:00-4:00 AM UTC
"""

FAQ = """FREQUENTLY ASKED QUESTIONS (FAQ)
TechCorp Customer Support FAQ - Updated June 2024

Q1: How do I reset my password?
A: Go to app.techcorp.com/reset-password and enter your email address.
   You will receive a reset link within 2 minutes. Check your spam folder.
   The link expires after 24 hours.

Q2: What happens when my free trial ends?
A: Your account pauses automatically and no charges occur. You have 7 days
   to choose a plan. After 7 days, your data is preserved for 30 more days
   before deletion. You can reactivate anytime within that window.

Q3: Can I upgrade or downgrade my plan at any time?
A: Yes. Upgrades take effect immediately (prorated charge applies).
   Downgrades take effect at the end of your current billing cycle.

Q4: My application crashes when I upload a file. What should I do?
A: This is usually a file size or format issue. Check your plan upload limit
   (Starter: 25 MB, Pro: 100 MB). Supported formats: PDF, DOCX, XLSX, PNG,
   JPG, MP4, and ZIP. Clear browser cache and retry. For large files, use
   the desktop app instead of the browser.

Q5: How do I cancel my subscription?
A: Go to Settings > Billing > Cancel Subscription. Your access continues
   until end of the current billing period. Cancellation requests require
   supervisor approval and may take up to 24 hours to process.

Q6: Can I get a refund?
A: Refunds are available within 30 days of purchase. After 30 days, refunds
   are reviewed case-by-case. Submit a refund request through the support
   portal. Processing takes 5-7 business days. All refund requests require
   manager approval before processing.

Q7: How many users can I add?
A: Starter: up to 3 users. Professional: up to 25 users. Enterprise: unlimited.
   Additional user seats can be purchased as an add-on on Pro plans.

Q8: Is my data secure?
A: Yes. TechCorp uses AES-256 encryption at rest and TLS 1.3 in transit.
   We are SOC 2 Type II certified and GDPR compliant.

Q9: What integrations are available?
A: Starter: Slack, Google Drive. Professional: adds Salesforce, HubSpot, Zapier.
   Enterprise: Custom integrations available on request.

Q10: How do I contact support?
A: Live chat: app.techcorp.com
   Email: support@techcorp.com
   Phone (Enterprise only): +1-800-TECHCORP
   Support hours: Monday-Friday 9 AM - 6 PM EST

Q11: What is the difference between pausing and closing my account?
A: Pausing: Data preserved, access suspended, can reactivate anytime.
   Closing: Permanent. All data deleted within 30 days. Cannot be undone.
   Account closure requires senior support approval.

Q12: Can I use TechCorp offline?
A: The desktop app supports limited offline access for cached content.
   Changes sync automatically when you reconnect to the internet.
"""


def write_pdf(filename: str, content: str, title: str) -> None:
    """
    Create a properly formatted PDF using the modern fpdf2 API.

    Uses fpdf2 >= 2.5.2 API:
      - new_x / new_y instead of deprecated ln=True
      - Explicit margins for reliable multi_cell rendering
      - Encodes content to latin-1 safe characters
    """
    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(
        0, 10, title,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C"
    )
    pdf.ln(4)

    # Body — write each line individually
    pdf.set_font("Helvetica", size=9)
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin  # usable page width

    for line in content.strip().split("\n"):
        # Sanitize: replace non-latin-1 characters to avoid encoding errors
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(effective_width, 5, safe_line)

    filepath = os.path.join(DOCS_DIR, filename)
    pdf.output(filepath)
    size_kb = os.path.getsize(filepath) // 1024
    print(f"  Created: {filename} ({size_kb} KB)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Creating company documents for RAG system")
    print("=" * 60 + "\n")

    documents = [
        ("company_policy.pdf",   COMPANY_POLICY,   "TechCorp Company Policy"),
        ("pricing_guide.pdf",    PRICING_GUIDE,    "TechCorp Pricing Guide"),
        ("technical_manual.pdf", TECHNICAL_MANUAL, "TechCorp Technical Manual"),
        ("faq.pdf",              FAQ,              "TechCorp FAQ"),
    ]

    for filename, content, title in documents:
        write_pdf(filename, content, title)

    print(f"\nAll 4 documents created in: {DOCS_DIR}")
    print("Ready for RAG ingestion in the next module.\n")
