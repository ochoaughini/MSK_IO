import base64
import hmac
import hashlib
import time
from typing import Dict, Optional


class SMSGateway:
    """Stub SMS gateway interface."""

    def send_sms(self, to: str, message: str) -> bool:
        print(f"Simulating SMS send to {to}: {message}")
        return True

    def receive_sms(self, from_num: str, message: str) -> Dict[str, str]:
        return {"from": from_num, "body": message}


gateway = SMSGateway()

SECRET_KEY = b"secret_key"


def encode_token(token: str) -> str:
    timestamp = str(int(time.time()))
    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
    data = f"{timestamp}:{token}:{token_hash}"
    hmac_sig = hmac.new(SECRET_KEY, data.encode(), hashlib.sha256).hexdigest()[:32]
    encoded = base64.b32encode(f"{data}:{hmac_sig}".encode()).decode()
    return encoded[:140]


def send_sms_token(to: str, token: str) -> bool:
    encoded = encode_token(token)
    return gateway.send_sms(to, encoded)


def parse_and_validate_sms(message: str) -> Optional[str]:
    try:
        decoded = base64.b32decode(message.encode()).decode()
        timestamp, token, token_hash, received_hmac = decoded.split(":")
        data = f"{timestamp}:{token}:{token_hash}"
        expected_hmac = hmac.new(SECRET_KEY, data.encode(), hashlib.sha256).hexdigest()[:32]
        if received_hmac != expected_hmac:
            return None
        if hashlib.sha256(token.encode()).hexdigest()[:16] != token_hash:
            return None
        if int(time.time()) - int(timestamp) > 3600:
            return None
        print(f"Token reconstructed and validated: {token}")
        return token
    except Exception:
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: sms_module.py send <phone> <token> | receive <message>")
        raise SystemExit(1)

    if sys.argv[1] == "send" and len(sys.argv) >= 4:
        send_sms_token(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "receive" and len(sys.argv) >= 3:
        token = parse_and_validate_sms(sys.argv[2])
        print(f"Validated token: {token}")
    else:
        print("Invalid arguments")
        raise SystemExit(1)
