"""Jarvis — Cloudflare Tunnel Integration.

Gerencia o túnel público e seguro do Cloudflare (cloudflared) para permitir
o acesso remoto ao painel web do Jarvis de forma simples e segura.
"""

from __future__ import annotations

import atexit
import re
import subprocess
import sys
import time
from pathlib import Path

from jarvis.core.logging import get_logger

log = get_logger(__name__)


def start_cloudflare_tunnel(
    port: int,
    ssl_enabled: bool,
    project_root: Path,
    token: str = "",
    domain: str = "",
) -> tuple[str | None, subprocess.Popen | None]:
    """Inicia o executável cloudflared e extrai a URL pública do túnel.

    Registra um hook atexit para garantir que o processo seja finalizado quando o
    Jarvis encerrar. Supports tunnels run via token and domain.
    """
    binary_name = "cloudflared.exe" if sys.platform == "win32" else "cloudflared"
    cf_path = project_root / binary_name

    if not cf_path.exists():
        log.warning(f"Executável do Cloudflare ({binary_name}) não encontrado em: {cf_path}")
        return None, None

    if token:
        # Modo Túnel Fixo por Token
        cmd = [str(cf_path.resolve()), "tunnel", "run", "--token", token]
        log.info("Iniciando túnel Cloudflare fixo por token...")
    else:
        # Modo Túnel Rápido (Quick Tunnel) temporário
        protocol = "https" if ssl_enabled else "http"
        local_url = f"{protocol}://localhost:{port}"
        cmd = [str(cf_path.resolve()), "tunnel", "--url", local_url]
        if ssl_enabled:
            cmd.append("--no-tls-verify")
        log.info(f"Iniciando túnel Cloudflare temporário para {local_url}...")

    try:
        # Executa em background redirecionando stderr para stdout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        log.error(f"Falha ao executar cloudflared: {e}")
        return None, None

    # Registra o encerramento automático do processo ao sair do script
    def cleanup_tunnel():
        if process.poll() is None:
            log.info("Cloudflare: Encerrando túnel...")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            log.info("Cloudflare: Túnel encerrado com sucesso.")

    atexit.register(cleanup_tunnel)

    if token:
        # Se for um túnel fixo via token, não procuramos por URL temporária nos logs.
        # Retornamos o domínio se o usuário o informou.
        public_url = f"https://{domain}" if domain else "Túnel Fixo Ativo (verifique no painel do Cloudflare)"
        # Dorme um breve período para certificar de que o processo não morreu imediatamente por token inválido etc.
        time.sleep(1.5)
        if process.poll() is not None:
            error_output = process.stdout.read().strip()
            log.error(f"O processo cloudflared encerrou prematuramente com código: {process.returncode}")
            if error_output:
                log.error(f"Saída de erro do cloudflared:\n{error_output}")
            return None, process
        log.info(f"Túnel Cloudflare via token estabelecido! URL: {public_url}")
        return public_url, process

    # Expressão regular para encontrar a URL pública gerada no log do cloudflared
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    public_url = None

    # Monitora a saída linha por linha com timeout para evitar travamento eterno
    start_time = time.time()
    while time.time() - start_time < 12.0:  # Timeout de 12 segundos
        # Non-blocking check se o processo já morreu
        if process.poll() is not None:
            log.error(f"O processo cloudflared encerrou prematuramente com código: {process.returncode}")
            break

        line = process.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue

        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            log.info(f"Túnel Cloudflare estabelecido! URL pública: {public_url}")
            break

    if not public_url:
        log.warning("Não foi possível obter a URL pública do Cloudflare (timeout excedido).")

    return public_url, process
