"""Jarvis — Cliente REST do Home Assistant.

Permite consultar estados de dispositivos e invocar serviços do Home Assistant.
"""

from __future__ import annotations

import asyncio
import httpx
from jarvis.config.settings import get_settings
from jarvis.core.logging import get_logger

log = get_logger(__name__)


class HomeAssistantClient:
    """Cliente assíncrono para comunicação com o Home Assistant OS via REST API."""

    def __init__(self) -> None:
        self.settings = get_settings().ha
        self.url = self.settings.url.rstrip("/")
        self.token = self.settings.token
        self.entities: list[dict] = []  # Lista de entidades ativas formatadas

        # Cabeçalhos padrão para a API REST do Home Assistant
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    @property
    def is_configured(self) -> bool:
        """Verifica se a URL e o Token estão configurados no ambiente."""
        return bool(self.url and self.token and self.token != "seu_token_aqui")

    async def check_api(self) -> bool:
        """Verifica se a API do Home Assistant está online e acessível."""
        if not self.is_configured:
            log.warning("Home Assistant: URL ou Token não configurados no .env.")
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.url}/", headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    log.info("Home Assistant: Conexão bem-sucedida! Resposta: {}", data)
                    return True
                else:
                    log.error(
                        "Home Assistant: Falha ao conectar. HTTP Status: {}. Resposta: {}",
                        response.status_code,
                        response.text,
                    )
                    return False
        except Exception as e:
            log.error("Home Assistant: Erro de rede ao verificar API: {}", e)
            return False

    async def get_entities(self) -> list[dict]:
        """Consulta o estado de todas as entidades do Home Assistant.

        Retorna:
            Uma lista de dicionários contendo:
            - entity_id: ID único da entidade (ex: 'light.living_room')
            - name: Nome amigável do dispositivo (ex: 'Luz da Sala')
            - state: Estado atual (ex: 'on', 'off', 'unavailable')
        """
        if not self.is_configured:
            return []

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.url}/states", headers=self.headers)
                response.raise_for_status()
                
                raw_states = response.json()
                synced_entities = []
                for item in raw_states:
                    entity_id = item.get("entity_id", "")
                    state = item.get("state", "unknown")
                    attributes = item.get("attributes", {})
                    friendly_name = attributes.get("friendly_name", entity_id)

                    # Ignora zonas, automações complexas sem nome legível, etc., para reduzir ruído no prompt
                    if not entity_id or not friendly_name:
                        continue

                    # Filtra apenas domínios controláveis úteis para controle de voz
                    domain = entity_id.split(".")[0]
                    if domain not in ("light", "switch", "climate", "media_player", "fan", "cover", "lock"):
                        continue

                    # Vamos guardar as informações essenciais
                    synced_entities.append({
                        "entity_id": entity_id,
                        "name": friendly_name,
                        "state": state,
                    })

                self.entities = synced_entities
                log.info("Home Assistant: Sincronizadas {} entidades com sucesso.", len(self.entities))
                return self.entities

        except Exception as e:
            log.error("Home Assistant: Falha ao carregar estados das entidades: {}", e)
            return []

    async def control_entity(
        self,
        domain: str,
        service: str,
        entity_id: str,
        data: dict | None = None,
        delay: float = 0.0,
    ) -> dict | None:
        """Invoca um serviço para controlar um dispositivo no Home Assistant.

        Args:
            domain: Domínio da entidade (ex: 'light', 'switch', 'climate')
            service: Nome do serviço a invocar (ex: 'turn_on', 'turn_off', 'set_temperature')
            entity_id: ID da entidade-alvo (ex: 'light.living_room')
            data: Parâmetros adicionais opcionais (ex: {"temperature": 22.0})

        Retorna:
            Dicionário com o resultado retornado pelo Home Assistant ou None se falhar.
        """
        if not self.is_configured:
            log.warning("Home Assistant: Tentativa de controle sem credenciais válidas.")
            return None

        if delay > 0:
            log.info("Home Assistant: Aguardando {} segundos antes de executar o serviço na entidade {}...", delay, entity_id)
            await asyncio.sleep(delay)

        payload = {"entity_id": entity_id}
        if data:
            payload.update(data)

        url = f"{self.url}/services/{domain}/{service}"
        log.info("Home Assistant: Chamando serviço {} no domínio {} para a entidade {} com payload: {}", service, domain, entity_id, payload)

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()
                log.info("Home Assistant: Serviço executado com sucesso.")
                return result
        except httpx.HTTPStatusError as e:
            log.error(
                "Home Assistant: Falha HTTP ao executar serviço. Status: {}. Resposta: {}",
                e.response.status_code,
                e.response.text,
            )
            return None
        except Exception as e:
            log.error("Home Assistant: Erro ao executar serviço para a entidade {}: {}", entity_id, e)
            return None
