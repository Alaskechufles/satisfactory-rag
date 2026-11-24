# Alcance de satisrag

Este documento decide qué entra en el backlog y cómo se mide si el asistente funciona. Si una idea no ayuda a la persona objetivo con uno de los cuatro tipos de pregunta, no se construye en la v1.

## Persona objetivo

**Jugador en sus primeras 20 horas, entre tier 0 y tier 4.** Juega en español y pregunta mientras juega, con frases cortas y sin el vocabulario exacto ("placas", "varillas", "la máquina de tornillos"). Necesita una respuesta que pueda aplicar enseguida, con los nombres tal como los verá en el juego y en la wiki.

**Cobertura:** se indexa todo el juego, pero la calidad se promete y se mide solo en tiers 0–4. Las preguntas de tiers superiores se responden sin esa garantía.

## Tipos de pregunta

Los ejemplos usan recetas estándar y están verificados con los datos del juego. Ilustran el alcance y **no deben copiarse al dataset de evaluación**: así el bot no se diseña mirando el examen.

### 1. Conceptual

Qué es algo, para qué sirve o en qué se diferencia de otra cosa. Se resuelve con RAG.

| Pregunta | Respuesta esperada |
| --- | --- |
| ¿Para qué sirve la Ensambladora (Assembler)? | Combina dos piezas distintas para fabricar otra. |
| ¿En qué se diferencian el Horno (Smelter) y la Fundición (Foundry)? | El Horno convierte mineral en lingotes; la Fundición funde dos recursos en lingotes de aleación. |
| ¿Qué hace un Divisor de cinta transportadora (Conveyor Splitter)? | Reparte una cinta en dos o tres. |

### 2. Consulta exacta

Un dato concreto de una receta, un edificio o un coste. Se resuelve con `get_recipe` u otra consulta sobre el modelo de dominio; RAG aporta la explicación.

| Pregunta | Respuesta esperada |
| --- | --- |
| ¿Qué necesito para fabricar un Rotor? | 5 Barras de hierro (Iron Rod) y 25 Tornillos (Screws), en una Ensambladora, cada 15 s. |
| ¿Qué recetas alternativas hay para los Tornillos (Screws)? | Dos: Tornillos fundidos (Cast Screws) y Tornillos de acero (Steel Screws). |
| ¿Cuánto cuesta el hito Energía del carbón (Coal Power)? | 150 Planchas de hierro reforzadas (Reinforced Iron Plate), 50 Rotores (Rotor) y 500 Cables (Cable). |

### 3. Cuantitativa

Cuántas máquinas o recursos hacen falta para una tasa de producción. La calcula Python con `production_calculator`; el LLM solo la explica.

| Pregunta | Respuesta esperada |
| --- | --- |
| ¿Cuántos Constructores para 60 Planchas de hierro (Iron Plate) por minuto? | 60 ÷ 20 = **3 Constructores**. |
| ¿Cuántas Ensambladoras para 10 Rotores por minuto? | 10 ÷ 4 = 2,5 → **3 Ensambladoras**. |
| ¿Qué máquinas necesito para 10 Planchas de hierro reforzadas por minuto, desde el mineral? | **2 Ensambladoras, 8 Constructores y 4 Hornos** (desglose abajo). |

Desglose del tercer ejemplo:

| Pieza | Tasa necesaria | Tasa por máquina | Máquinas |
| --- | --- | --- | --- |
| Plancha de hierro reforzada (Reinforced Iron Plate) | 10/min | 5/min | 2 Ensambladoras |
| Plancha de hierro (Iron Plate) | 60/min | 20/min | 3 Constructores |
| Tornillos (Screws) | 120/min | 40/min | 3 Constructores |
| Barra de hierro (Iron Rod) | 30/min | 15/min | 2 Constructores |
| Lingote de hierro (Iron Ingot) | 120/min | 30/min | 4 Hornos |
| Mineral de hierro (Iron Ore) | 120/min | – | – |

### 4. Guía

En qué orden progresar y qué hace falta para llegar a algo. En la v1 se limita a lo que cubren los hitos y las mejoras del HUB en los datos del juego. Se resuelve con RAG y filtro por tier.

| Pregunta | Respuesta esperada |
| --- | --- |
| ¿Cómo consigo el Constructor? | Con la Mejora del HUB 3 (HUB Upgrade 3), después de completar la 1 y la 2. |
| ¿Qué puedo desbloquear en el tier 1? | Construcción de base (Base Building), Logística (Logistics) e Investigación de campo (Field Research). |
| ¿Qué necesito antes de producir acero? | El hito Producción de acero básica (Basic Steel Production), de tier 3. Pide Rotores y Armazones modulares (Modular Frame), así que antes hace falta Montaje de piezas (Part Assembly), de tier 2. |

## Fuera de alcance

El asistente dice que no puede responder, no inventa datos y, si puede, indica qué sí sabe hacer.

**Excluido del proyecto:**

- Mods.
- Speedrun.
- Lore.
- Mapas y coordenadas de recursos.
- Diseño físico de fábricas. El bot explica y calcula, pero no diseña.
- Cualquier tema ajeno a Satisfactory.

**Aplazado, fuera de la v1:**

- Cálculos con overclock.
- Consejos de estrategia, hasta que se indexen guías de la wiki (F1.4).

## Reglas

### Nombres e idioma

- El bot responde en español y cita el nombre oficial con el inglés entre paréntesis: Plancha de hierro (Iron Plate).
- Los nombres oficiales salen de `es-ES.json` y `en-US.json`.
- Los términos habituales se guardan como sinónimos: "placa" → Plancha, "varilla" → Barra, "Ensamblador" → Ensambladora.
- Un sinónimo nunca sustituye a un nombre oficial. Por ejemplo, Placas inteligentes (Smart Plating) ya usa "placa", y Fundición es Foundry, no Smelter.

### Recetas

- Las alternativas están en los datos y en las consultas.
- La calculadora usa la receta estándar, salvo que el jugador nombre otra.

### Cálculo

- El LLM nunca hace aritmética.
- El número de máquinas es un entero, redondeado hacia arriba en cada receta de la cadena.
- La demanda se propaga con tasas exactas, sin redondear: 2,5 Ensambladoras consumen por 2,5, aunque se construyan 3.
- Se supone velocidad de reloj del 100%.

## Métricas objetivo

| Tipo | Preguntas | Métrica | Umbral | Fallos permitidos | Se mide desde |
| --- | --- | --- | --- | --- | --- |
| Conceptual | 9 | recall@5, junto con consulta | ≥ 0,85 | 3 de 20* | S4 |
| Consulta exacta | 11 | recall@5, junto con conceptual | ≥ 0,85 | 3 de 20* | S4 |
| Cuantitativa | 9 | Exactitud numérica | 100% | 0 | S7 |
| Guía | 7 | recall@5 | ≥ 0,70 | 2 de 7* | S4 |
| Fuera de alcance | 4 | Rechazo correcto | 100% | 0 | S6 |
| Respondidas con RAG | – | Fidelidad | ≥ 0,9 | – | S7 |

\* Si cada pregunta espera una sola entidad.

**Definiciones**

- **recall@5:** proporción de entidades esperadas que aparecen entre los 5 primeros resultados, promediada entre preguntas.
- **Exactitud numérica:** todos los números esperados coinciden. Las máquinas, exactas; las tasas por minuto, con tolerancia de ±0,01.
- **Rechazo correcto:** la respuesta reconoce que no puede responder o redirige, y no inventa datos.
- **Fidelidad:** proporción de afirmaciones respaldadas por el contexto recuperado o por el resultado de una herramienta.

**Cómo leerlas**

- El dataset es pequeño. Entre conceptuales y consultas, un solo fallo mueve la métrica 5 puntos. Al comparar configuraciones se revisa también qué preguntas cambian.
- El rechazo correcto no se lee solo: un bot que lo rechaza todo sacaría 100%. Debe cumplirse a la vez que las demás métricas.

## Pendiente para la Épica 1

- **Idioma de las descripciones.** Algunas traducciones invierten el significado: la Ensambladora "Crafts 2 parts into another part" aparece como "Fabrica 2 piezas a partir de otra". Hay que decidir si las descripciones se toman del inglés y solo los nombres del español.
- **Versión del juego.** Confirmar la versión exacta de los JSON para fijar `game_version`.
