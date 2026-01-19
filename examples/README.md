# Exemples d'utilisation de langchain-copilot

Ce dossier contient plusieurs exemples montrant comment utiliser `langchain-copilot`.

## Liste des exemples

1. **[01_simple_invoke.py](01_simple_invoke.py)** - Invocation synchrone simple
2. **[02_streaming.py](02_streaming.py)** - Réponse en streaming
3. **[03_async_invoke.py](03_async_invoke.py)** - Invocation asynchrone
4. **[04_async_streaming.py](04_async_streaming.py)** - Streaming asynchrone
5. **[05_langchain_chain.py](05_langchain_chain.py)** - Utilisation dans une chaîne LangChain
6. **[06_temperature.py](06_temperature.py)** - Contrôle de la température

## Comment exécuter les exemples

Chaque exemple peut être exécuté indépendamment :

```bash
# Exemple d'invocation simple
python examples/01_simple_invoke.py

# Exemple de streaming
python examples/02_streaming.py

# Exemple asynchrone
python examples/03_async_invoke.py

# Etc.
```

Ou exécutez tous les exemples à la fois avec `basic_usage.py` :

```bash
python examples/basic_usage.py
```
