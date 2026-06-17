"""Jarvis — SSL certificate generation utility.

Generates self-signed SSL certificates automatically if they don't exist.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from loguru import logger


def generate_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    """Gera um certificado e chave privada SSL autoassinados caso não existam."""
    if cert_path.exists() and key_path.exists():
        logger.info(f"Certificados SSL já existem em: {cert_path.parent}")
        return

    logger.info("Gerando certificados SSL autoassinados para conexão HTTPS segura...")
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        # 1. Gera chave privada RSA
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # 2. Configura a identificação do sujeito e emissor
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "BR"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "SP"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Sao Paulo"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Jarvis Local"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])

        # 3. Constrói o certificado autoassinado válido por 1 ano
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("127.0.0.1"),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        # 4. Grava a chave privada PEM
        key_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

        # 5. Grava o certificado PEM
        cert_path.write_bytes(
            cert.public_bytes(serialization.Encoding.PEM)
        )
        logger.info(f"Certificados SSL gerados com sucesso na pasta: {cert_path.parent}")

    except Exception as e:
        logger.error(f"Falha ao gerar certificados SSL programáticos usando 'cryptography': {e}")
        logger.info("Tentando fallback para comando do sistema OpenSSL...")
        try:
            import subprocess
            cmd = [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path), "-out", str(cert_path),
                "-days", "365", "-nodes",
                "-subj", "/C=BR/ST=SP/L=SaoPaulo/O=JarvisLocal/CN=localhost"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Certificados SSL gerados com sucesso via OpenSSL em: {cert_path.parent}")
        except Exception as openssl_err:
            logger.critical(f"Falha crítica no fallback do OpenSSL: {openssl_err}")
            raise RuntimeError(
                "Não foi possível gerar os certificados SSL necessários. "
                "Certifique-se de que a biblioteca 'cryptography' está funcional ou que o OpenSSL está instalado no sistema."
            ) from openssl_err
