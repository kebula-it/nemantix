import smtplib
import imaplib
import ssl
import email
import os
import json

from email.header import decode_header
from typing import List, Dict, Optional, Any

from nemantix.core import tool, Toolset


class EmailToolset(Toolset):
    """
    Initializes the EmailToolset with necessary credentials and server configurations.

    Args:
        email_user (str, optional): The email address used for authentication. If None, tries 'EMAIL_USER' env var.
        email_password (str, optional): The password or app-specific password. If None, tries 'EMAIL_PASSWORD' env var.
        smtp_server (str, optional): The SMTP server address. Defaults to "smtp.gmail.com".
        smtp_port (int, optional): The SMTP port (465 for SSL, 587 for STARTTLS). Defaults to 465.
        imap_server (str, optional): The IMAP server address. Defaults to "imap.gmail.com".
        imap_port (int, optional): The IMAP port. Defaults to 993.

    Example call:
        EmailToolset(
            email_user="user@example.com",
            email_password="secret_password",
            smtp_port=587
        )
    """

    def __init__(
            self,
            email_user: Optional[str] = None,
            email_password: Optional[str] = None,
            smtp_server: str = "smtp.gmail.com",
            smtp_port: int = 465,
            imap_server: str = "imap.gmail.com",
            imap_port: int = 993,
    ):
        # 1. Try arguments first, then environment variables
        super().__init__()
        self.email_user = email_user or os.getenv("EMAIL_USER")
        self.password = email_password or os.getenv("EMAIL_PASSWORD")

        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.imap_server = imap_server
        self.imap_port = imap_port

    @classmethod
    def from_config(cls, config_path: str) -> "EmailToolset":
        """
        Factory method to create an instance from a JSON configuration file.

        Args:
            config_path (str): The file path to the JSON configuration file containing credentials.

        Returns:
            EmailToolset: An initialized instance of the EmailToolset class.

        Example call:
            EmailToolset.from_config(
                config_path="email_config.json"
            )
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Unpack the dictionary as arguments to __init__
        return cls(**config)

    @tool
    def send_email(self, recipient: str, subject: str, body: str) -> str:
        """
        Sends an email via SMTP (Supports SSL and STARTTLS).

        Args:
            recipient (str): The email address of the recipient.
            subject (str): The subject line of the email.
            body (str): The plain text content of the email body.

        Returns:
            str: A message indicating success or failure of the operation.

        Example call:
            send_email(
                recipient="gioele.ciaparrone@kebula.it",
                subject="Test send",
                body="This is an automatically sent test email."
            )
        """
        if not self.email_user or not self.password:
            return "Error: Credentials not set."

        msg = email.message.EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = self.email_user
        msg["To"] = recipient

        try:
            context = ssl.create_default_context()

            # Port 465 = Implicit SSL (Gmail/Yahoo)
            if self.smtp_port == 465:
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                    server.login(self.email_user, self.password)
                    server.send_message(msg)

            # Port 587 = STARTTLS (Outlook/Office 365)
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.email_user, self.password)
                    server.send_message(msg)

            return f"Success: Email sent to {recipient}."
        except Exception as e:
            return f"Failed to send email: {str(e)}"

    @tool
    def read_emails(self, limit: int = 5, folder: str = "INBOX") -> List[Dict[str, Any]]:
        """
        Reads recent emails via IMAP.

        Args:
            limit (int, optional): The maximum number of recent emails to fetch. Defaults to 5.
            folder (str, optional): The mailbox folder to search in (e.g., "INBOX", "SENT"). Defaults to "INBOX".

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents an email containing 'from', 'subject', and 'body_preview' keys.

        Example call:
            read_emails(
                limit=3,
                folder="INBOX"
            )
        """
        if not self.email_user or not self.password:
            return [{"error": "Credentials not set."}]

        try:
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            mail.login(self.email_user, self.password)

            status, _ = mail.select(folder)
            if status != "OK":
                return [{"error": f"Could not select folder: {folder}"}]

            status, data = mail.search(None, "ALL")
            mail_ids = data[0].split()

            # Fetch most recent 'limit' emails
            latest_ids = mail_ids[-limit:]
            results = []

            for i in reversed(latest_ids):
                _, msg_data = mail.fetch(i, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])

                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding if encoding else "utf-8")

                        # Simplified body extraction logic
                        body = "(No plain text body found)"
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    payload = part.get_payload(decode=True)
                                    if payload and isinstance(payload, bytes):
                                        body = payload.decode()
                                    break
                        else:
                            payload = msg.get_payload(decode=True)
                            if payload and isinstance(payload, bytes):
                                body = payload.decode()

                        results.append({
                            "from": msg.get("From"),
                            "subject": subject,
                            "body_preview": body[:100] + "...",
                        })

            mail.logout()
            return results

        except Exception as e:
            return [{"error": f"Failed to read emails: {str(e)}"}]
