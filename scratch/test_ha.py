import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Garante que a pasta src está no path
sys.path.insert(0, "src")

from unittest.mock import Mock, AsyncMock, patch
from jarvis.core.homeassistant import HomeAssistantClient
from jarvis.config.settings import get_settings


async def run_mock_tests():
    print("=== Executando Testes MOCK da Integração Home Assistant ===")
    
    # Mock para verificar comportamento offline e respostas esperadas
    client = HomeAssistantClient()
    client.url = "http://mock-homeassistant:8123/api"
    client.token = "mocked_valid_token"
    
    # 1. Testar get_entities mockado
    mock_states = [
        {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"friendly_name": "Luz da Sala"}
        },
        {
            "entity_id": "switch.coffee_maker",
            "state": "off",
            "attributes": {"friendly_name": "Cafeteira"}
        }
    ]
    
    with patch("httpx.AsyncClient.get") as mock_get:
        # Mock do retorno da API
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = lambda: mock_states
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response
        
        entities = await client.get_entities()
        print(f"Entities retornadas: {entities}")
        assert len(entities) == 2
        assert entities[0]["entity_id"] == "light.living_room"
        assert entities[0]["name"] == "Luz da Sala"
        assert entities[0]["state"] == "on"
        print("✅ Teste Mock de get_entities passou com sucesso!")

    # 2. Testar control_entity mockado
    mock_post_res = {"context": {"id": "test_id"}}
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = lambda: mock_post_res
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response
        
        result = await client.control_entity("light", "turn_on", "light.living_room")
        print(f"Resultado do controle: {result}")
        assert result == mock_post_res
        print("✅ Teste Mock de control_entity passou com sucesso!")


async def run_real_test():
    print("\n=== Executando Teste REAL com Credenciais Locais ===")
    client = HomeAssistantClient()
    
    if not client.is_configured:
        print("⚠️ Home Assistant não configurado no .env (token/URL ausente). Pulando teste real.")
        return
        
    print(f"URL: {client.url}")
    print("Verificando conexão...")
    is_online = await client.check_api()
    
    if not is_online:
        print("❌ Home Assistant está configurado mas a conexão falhou. Verifique se o servidor está online.")
        return
        
    print("Sincronizando entidades...")
    entities = await client.get_entities()
    print(f"Entidades encontradas ({len(entities)}):")
    for e in entities[:10]: # Mostra as 10 primeiras
        print(f" - {e['name']} ({e['entity_id']}) -> Estado: {e['state']}")
        
    print("\n✅ Conexão e sincronização real concluídas!")


async def main():
    await run_mock_tests()
    await run_real_test()


if __name__ == "__main__":
    asyncio.run(main())
