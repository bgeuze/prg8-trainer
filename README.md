# Hand Gesture Training Tool

Met deze tool train je je eigen handgebaren voor de quiz-app. Je maakt voor elke letter (A, B, C, D) een uniek gebaar, verzamelt voorbeelden, en slaat het model op als JSON.

## Features

- Train je eigen handgebaren voor A, B, C, D
- Zie direct hoeveel voorbeelden je per label hebt
- Evalueer je model (accuracy, confusion matrix)
- Sla je model op als JSON voor gebruik in de hoofdapp

## Installatie & Gebruik

1. **Start een lokale webserver** in de hoofdmap:
   ```bash
   python3 -m http.server 8000
   ```
   Of gebruik de Live Server extensie in VSCode.

2. **Open `index.html` in je browser** via de lokale server.

3. **Train je model:**
   - Klik op een letter-knop (A, B, C, D) en maak het bijbehorende handgebaar voor de camera.
   - Herhaal voor elk label tot je minimaal 20 voorbeelden per label hebt.
   - Evalueer je model en sla het op.

4. **Gebruik het model in de hoofdapp.**

## Benodigdheden

- Een moderne browser (Chrome, Edge, Firefox, Safari)
- Webcam

## Licentie

MIT
