Vehicle:
  generates benign messages
  → runs IDS
  → trains model

Attacker:
  generates malicious messages
  → poisons weights

Server:
  receives both
  → Krum filters attacker
  → updates global model