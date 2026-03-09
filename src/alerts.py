"""
alerts.py — Email (Gmail SMTP) and WhatsApp (Twilio) alert modules.
"""

import smtplib
import ssl
from email.mime.text   import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# ── Email Alert ────────────────────────────────────────────────────────────────

def send_email_alert(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str,
    body_html: str,
) -> dict:
    """
    Send an HTML email alert via SMTP (Gmail-compatible).
    Returns {"success": True} or {"success": False, "error": str}.
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender_email
        msg["To"]      = recipient_email

        part = MIMEText(body_html, "html")
        msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_fraud_email(transaction_data: dict, fraud_score: float, risk_level: str) -> str:
    """Build a styled HTML email body for a fraud alert."""
    color = {"CRITICAL": "#e63946", "HIGH": "#ff6b35", "MEDIUM": "#ffd700", "LOW": "#06d6a0"}.get(risk_level, "#e63946")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = "".join(
        f"<tr><td style='padding:6px 12px;color:#888;'>{k}</td>"
        f"<td style='padding:6px 12px;color:#fff;font-weight:600;'>{v}</td></tr>"
        for k, v in transaction_data.items()
    )
    return f"""
    <html><body style="background:#0a0a1a;font-family:Inter,sans-serif;padding:32px;">
      <div style="max-width:600px;margin:auto;background:#12122a;border-radius:16px;
                  border:1px solid #2a2a4a;overflow:hidden;">
        <div style="background:linear-gradient(135deg,{color},#0a0a1a);padding:24px 32px;">
          <h1 style="color:#fff;margin:0;font-size:22px;">🛡️ FraudShield AI — Fraud Alert</h1>
          <p style="color:rgba(255,255,255,.7);margin:4px 0 0;">Detected at {ts}</p>
        </div>
        <div style="padding:24px 32px;">
          <div style="background:{color}22;border:1px solid {color};border-left:4px solid {color};
                      border-radius:8px;padding:16px;margin-bottom:20px;">
            <span style="color:{color};font-size:18px;font-weight:800;">⚠️ Risk Level: {risk_level}</span><br>
            <span style="color:#f0f0f0;font-size:28px;font-weight:900;">{fraud_score:.1%} Fraud Probability</span>
          </div>
          <h3 style="color:#c8c8e0;margin:0 0 12px;">Transaction Details</h3>
          <table style="width:100%;border-collapse:collapse;background:#0d0d20;border-radius:8px;overflow:hidden;">
            {rows}
          </table>
          <p style="color:#4a4a6a;font-size:12px;margin-top:24px;text-align:center;">
            FraudShield AI &nbsp;·&nbsp; Automated Fraud Detection System &nbsp;·&nbsp; Do not reply to this email
          </p>
        </div>
      </div>
    </body></html>
    """


# ── WhatsApp Alert via Twilio ─────────────────────────────────────────────────

def send_whatsapp_alert(
    account_sid: str,
    auth_token: str,
    from_number: str,   # e.g. "whatsapp:+14155238886"  (Twilio sandbox)
    to_number: str,     # e.g. "whatsapp:+91XXXXXXXXXX"
    fraud_score: float,
    risk_level: str,
    amount: str,
    transaction_id: str = "N/A",
    template: str = None,
) -> dict:
    """
    Send a WhatsApp message via Twilio API.
    If `template` is provided, placeholders are substituted; otherwise a default message is used.
    Returns {"success": True, "sid": str} or {"success": False, "error": str}.
    """
    try:
        from twilio.rest import Client
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Auto-add whatsapp: prefix if missing
        if not from_number.startswith("whatsapp:"):
            from_number = "whatsapp:" + from_number
        if not to_number.startswith("whatsapp:"):
            to_number = "whatsapp:" + to_number

        DEFAULT_TEMPLATE = (
            "🚨 *FraudShield AI — Fraud Alert*\n\n"
            "⚠️ Risk Level: *{risk_level}*\n"
            "📊 Fraud Score: *{fraud_score}*\n"
            "💰 Est. Amount: *{amount}*\n"
            "🆔 Transaction: *{transaction_id}*\n"
            "🕐 Time: {time}\n\n"
            "Please log into FraudShield AI to review this transaction immediately.\n"
            "— FraudShield AI Monitor 🛡️"
        )

        body = (template or DEFAULT_TEMPLATE)\
            .replace("{risk_level}",    risk_level)\
            .replace("{fraud_score}",   f"{fraud_score:.1%}")\
            .replace("{amount}",        amount)\
            .replace("{transaction_id}", transaction_id)\
            .replace("{time}",          ts)

        client  = Client(account_sid, auth_token)
        message = client.messages.create(body=body, from_=from_number, to=to_number)
        return {"success": True, "sid": message.sid}
    except ImportError:
        return {"success": False, "error": "Twilio not installed. Run: pip install twilio"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Alert Summary Builder ─────────────────────────────────────────────────────

def build_alert_summary(n_critical: int, n_high: int, total_flagged: int, amount_at_risk: str) -> str:
    """Build a summary email body for batch fraud report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <html><body style="background:#0a0a1a;font-family:Inter,sans-serif;padding:32px;">
      <div style="max-width:600px;margin:auto;background:#12122a;border-radius:16px;
                  border:1px solid #2a2a4a;overflow:hidden;">
        <div style="background:linear-gradient(135deg,#e63946,#0a0a1a);padding:24px 32px;">
          <h1 style="color:#fff;margin:0;font-size:22px;">🛡️ FraudShield AI — Daily Report</h1>
          <p style="color:rgba(255,255,255,.7);margin:4px 0 0;">{ts}</p>
        </div>
        <div style="padding:24px 32px;">
          <table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">
            <tr style="background:#1a1a35;">
              <td style="padding:14px 16px;color:#e63946;font-weight:700;">🔴 CRITICAL Alerts</td>
              <td style="padding:14px 16px;color:#fff;font-size:20px;font-weight:800;">{n_critical}</td>
            </tr>
            <tr style="background:#0d0d20;">
              <td style="padding:14px 16px;color:#ff6b35;font-weight:700;">🟠 HIGH Alerts</td>
              <td style="padding:14px 16px;color:#fff;font-size:20px;font-weight:800;">{n_high}</td>
            </tr>
            <tr style="background:#1a1a35;">
              <td style="padding:14px 16px;color:#8888aa;font-weight:700;">📋 Total Flagged</td>
              <td style="padding:14px 16px;color:#fff;font-size:20px;font-weight:800;">{total_flagged}</td>
            </tr>
            <tr style="background:#0d0d20;">
              <td style="padding:14px 16px;color:#ffd700;font-weight:700;">💰 Amount at Risk</td>
              <td style="padding:14px 16px;color:#ffd700;font-size:20px;font-weight:800;">{amount_at_risk}</td>
            </tr>
          </table>
        </div>
      </div>
    </body></html>
    """
