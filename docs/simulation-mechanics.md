# Astar Island - Simulation Mechanics

## Map Generation

Maps are **procedurally generated** from a visible seed containing:
- **Ocean** borders surrounding the perimeter
- **Fjords** cutting inland from random edges
- **Mountain chains** formed through random walks
- **Forest patches** covering land in clustered groves
- **Initial settlements** placed on land, spaced apart

Players can reconstruct initial terrain layouts locally using the seed.

## Terrain Types

8 terrain types in the simulator map to 6 prediction classes:

| Prediction Class | Terrain Types | Behavior |
|-----------------|---------------|----------|
| 0 - Empty | Ocean, Plains | Ocean is impassable; plains are buildable flat land |
| 1 - Settlement | Active settlements | Dynamic — grow, shrink, get destroyed |
| 2 - Port | Coastal harbors | Settlements that develop port capability |
| 3 - Ruin | Collapsed settlements | Former settlements that were destroyed |
| 4 - Forest | Forest | Mostly static; reclaims ruined/abandoned land |
| 5 - Mountain | Mountain | Impassable, **never changes** |

## 50-Year Simulation Lifecycle

Each year cycles through **5 phases in order**:

### Phase 1: Growth
- Settlements produce food based on **adjacent terrain**
- Prosperous settlements grow in population
- Develop coastal ports and build longships for naval operations
- Expand by **founding new settlements** on nearby land

### Phase 2: Conflict
- Settlements **raid each other**
- Longships extend raiding range significantly
- Desperate settlements (low food) raid **more aggressively**
- Successful raids loot resources and damage defenders
- Conquered settlements sometimes **change faction allegiance**

### Phase 3: Trade
- Ports within range exchange goods if **not at war**
- Trading generates wealth and food for both parties
- Diffuses technology between partners

### Phase 4: Winter
- All settlements **lose food**
- Settlements collapse from:
  - Starvation
  - Sustained raids
  - Harsh winters
- Collapsed settlements become **ruins**
- Population disperses to nearby friendly settlements

### Phase 5: Environment
- Natural world slowly **reclaims abandoned land**
- Thriving settlements may **rebuild ruined sites** (new outposts inheriting resources and knowledge)
- Coastal ruins can become ports
- Unreclaimed ruins become forests or plains

## Settlement Properties

Each settlement tracks:
- Position (x, y)
- Population
- Food
- Wealth
- Defense
- Tech level
- Port status
- Longship ownership
- Faction allegiance (owner_id)

**Important**: Initial states expose only **position and port status**. Internal statistics (population, food, wealth, etc.) require simulation queries to access.

## Key Dynamics to Model

1. **Static terrain**: Mountains never change. Forests rarely change. Ocean never changes.
2. **Settlement lifecycle**: Founded -> grows -> possibly becomes port -> possibly raided -> possibly becomes ruin
3. **Ruin fate**: Ruin -> rebuilt by nearby settlement OR reclaimed as forest/plains
4. **Expansion patterns**: Settlements expand to nearby land cells
5. **Coastal advantage**: Ports enable trade and longship construction
6. **Starvation cascades**: Food shortages -> aggressive raiding -> more destruction -> more ruins
