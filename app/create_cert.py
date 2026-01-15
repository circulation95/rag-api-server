from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
import socket

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

def main():
    cn = "python-asyncua-client"
    hostname = socket.gethostname()  # 예: WANG

    # ✅ 둘 다 넣어버리면 mismatch 이슈가 사실상 끝납니다
    app_uris = [
        "urn:python:asyncua:client",
        "urn:example.org:FreeOpcUa:opcua-asyncio",
    ]

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LocalTest"),
    ])

    now = datetime.now(timezone.utc)

    san_items = [
        x509.DNSName("localhost"),
        x509.DNSName(hostname),              # ✅ WANG 들어감
        x509.IPAddress(ip_address("127.0.0.1")),
    ] + [x509.UniformResourceIdentifier(u) for u in app_uris]

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_items), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    with open("client_key.pem", "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open("client_cert.pem", "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print("✅ generated: client_cert.pem, client_key.pem")
    print("   CN =", cn)
    print("   Hostname DNS =", hostname)
    print("   URIs =", app_uris)

if __name__ == "__main__":
    main()
