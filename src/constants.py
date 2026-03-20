"""Named constants and tuning parameters for the Norse world simulator.

All numeric thresholds, limits, and configuration values live here.
No magic numbers in logic code -- reference these constants instead.
"""

# ---------------------------------------------------------------------------
# Map dimensions
# ---------------------------------------------------------------------------

DEFAULT_MAP_WIDTH = 40
DEFAULT_MAP_HEIGHT = 40
SIMULATION_YEARS = 50

# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

# Ocean border thickness (perimeter cells)
OCEAN_BORDER = 1

# Fjord generation
NUM_FJORDS_RANGE = (2, 5)  # min, max fjords per map
FJORD_LENGTH_RANGE = (5, 15)  # min, max length in cells
FJORD_WIDTH = 1  # cells wide

# Mountain generation
NUM_MOUNTAIN_CHAINS = 3
MOUNTAIN_CHAIN_LENGTH_RANGE = (4, 12)
MOUNTAIN_TURN_PROB = 0.3  # probability of changing direction during random walk

# Forest generation
NUM_FOREST_PATCHES = 8
FOREST_PATCH_SIZE_RANGE = (3, 8)  # cells per patch

# Settlement placement
NUM_INITIAL_SETTLEMENTS_RANGE = (4, 8)
MIN_SETTLEMENT_SPACING = 5  # minimum distance between settlements
INITIAL_POPULATION = 50
INITIAL_FOOD = 100
INITIAL_WEALTH = 0
INITIAL_DEFENSE = 10
INITIAL_TECH_LEVEL = 1

# ---------------------------------------------------------------------------
# Simulation: Growth phase
# ---------------------------------------------------------------------------

BASE_FOOD_PRODUCTION = 10
FOOD_PER_FOREST = 15
TECH_FOOD_BONUS = 2  # per tech level
GROWTH_FOOD_THRESHOLD_MULTIPLIER = 2  # food > pop * this -> growth
GROWTH_RATE = 0.1  # fraction of population added per growth tick
PORT_DEVELOPMENT_THRESHOLD = 80  # population needed to develop port
LONGSHIP_BUILD_THRESHOLD = 120  # population needed for longship
EXPANSION_POPULATION_THRESHOLD = 100  # population needed to found new settlement
EXPANSION_NEW_POPULATION = 20  # starting population of new settlement

# ---------------------------------------------------------------------------
# Simulation: Conflict phase
# ---------------------------------------------------------------------------

RAID_RANGE = 3  # base range for land raiding (Manhattan distance)
LONGSHIP_RAID_RANGE = 8  # extended range with longship
DESPERATE_RAID_BONUS = 1.5  # damage multiplier when desperate
RAID_LOOT_FRACTION = 0.3  # fraction of defender food/wealth looted
CONQUEST_PROB = 0.2  # probability of faction change on successful raid
RAID_DAMAGE_BASE = 15  # base damage per raid

# ---------------------------------------------------------------------------
# Simulation: Trade phase
# ---------------------------------------------------------------------------

TRADE_RANGE = 6  # max distance for port-to-port trade
TRADE_FOOD_EXCHANGE = 10  # food exchanged per trade
TRADE_WEALTH_GAIN = 5  # wealth gained per trade
TECH_DIFFUSION_PROB = 0.3  # probability of tech level spreading via trade

# ---------------------------------------------------------------------------
# Simulation: Winter phase
# ---------------------------------------------------------------------------

WINTER_SEVERITY_RANGE = (0.5, 1.5)  # min, max winter severity
FOOD_CONSUMPTION_RATE = 0.4  # fraction of population consumed as food
STARVATION_COLLAPSE_POP = 10  # collapse if pop < this and food <= 0
COLLAPSE_RAID_DAMAGE_MULTIPLIER = 2  # collapse if raid_damage > strength * this
REFUGEE_RANGE = 5  # distance refugees flee to friendly settlements
REFUGEE_FRACTION = 0.5  # fraction of pop that becomes refugees

# ---------------------------------------------------------------------------
# Simulation: Environment phase
# ---------------------------------------------------------------------------

RUIN_RECLAIM_AS_FOREST_PROB = 0.15  # probability ruin becomes forest per year
RUIN_REBUILD_PROB = 0.1  # probability thriving settlement rebuilds nearby ruin
RUIN_REBUILD_RANGE = 3  # max distance to rebuild a ruin
RUIN_TO_PLAINS_PROB = 0.05  # probability ruin becomes plains
REBUILD_INHERIT_TECH_FRACTION = 0.5  # fraction of parent tech inherited
REBUILD_POPULATION = 15  # starting population of rebuilt settlement

# ---------------------------------------------------------------------------
# Scoring (reference only -- actual scoring is server-side)
# ---------------------------------------------------------------------------

NUM_PREDICTION_CLASSES = 6
PROBABILITY_FLOOR = 0.01  # minimum probability to avoid infinite KL divergence
SCORE_DECAY_RATE = 3  # exp(-3 * weighted_kl)
SCORE_ENTROPY_THRESHOLD = 0.01  # cells with entropy below this are "static" (excluded from scoring)

# ---------------------------------------------------------------------------
# API and query budget
# ---------------------------------------------------------------------------

API_BASE_URL = "https://api.ainm.no"
TOTAL_QUERY_BUDGET = 50  # total queries per round across all seeds
QUERY_WARNING_THRESHOLD = 45  # warn when this many queries used
VIEWPORT_MIN_SIZE = 5  # minimum viewport dimension
VIEWPORT_MAX_SIZE = 15  # maximum viewport dimension
NUM_SEEDS = 5  # seeds per round
API_RETRY_MAX = 3  # max retries for transient failures
API_RETRY_BACKOFF = 1.0  # base delay in seconds for exponential backoff

# ---------------------------------------------------------------------------
# Prediction blending
# ---------------------------------------------------------------------------

OBSERVATION_WEIGHT = 0.8  # weight for server observations in blended predictions
SIMULATION_WEIGHT = 0.2  # weight for local Monte Carlo sim in blended predictions
STATIC_TERRAIN_CONFIDENCE = 0.99  # prob for known-static terrain (mountain, ocean)
DEFAULT_MC_RUNS = 100  # default number of Monte Carlo runs for prediction

# ---------------------------------------------------------------------------
# Query strategy
# ---------------------------------------------------------------------------

QUERIES_PER_SEED_COVERAGE = 8  # queries reserved for initial map coverage per seed
QUERIES_ADAPTIVE_RESERVE = 10  # queries reserved for adaptive follow-up across all seeds

# ---------------------------------------------------------------------------
# Query strategy: interest scoring
# ---------------------------------------------------------------------------

INTEREST_UNCOVERED_WEIGHT = 1.0
INTEREST_NEAR_SETTLEMENT_WEIGHT = 3.0
INTEREST_EXPANSION_ZONE_WEIGHT = 2.0
INTEREST_COASTAL_WEIGHT = 1.5
INTEREST_SETTLEMENT_RADIUS = 5
ADAPTIVE_VIEWPORT_MID_SIZE = 10

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

CALIBRATION_KL_SCALE = 8.0  # sigmoid scaling for KL -> obs weight mapping
CALIBRATION_BIAS_THRESHOLD = 0.05  # minimum delta to flag a bias as significant

# ---------------------------------------------------------------------------
# Observation smoothing
# ---------------------------------------------------------------------------

LAPLACE_ALPHA = 0.01  # Laplace smoothing pseudocount (low = trust observations more)
OBS_CONFIDENCE_K = 5  # confidence scaling: weight = count / (count + K)

# ---------------------------------------------------------------------------
# Terrain classification
# ---------------------------------------------------------------------------

NUM_INTERNAL_TYPES = 7  # number of internal terrain type categories
DIST_BIN_EDGES = [0, 1, 2, 3, 4, 5, 7, 10, 15, 999]  # per-distance bins for finer granularity

# ---------------------------------------------------------------------------
# Prior weighting
# ---------------------------------------------------------------------------

SURVIVE_WEIGHT = 2.0  # weight for survive rounds (balanced for collapse too)
COLLAPSE_WEIGHT = 1.0  # weight for rounds where settlements collapsed
SURVIVE_ROUNDS = frozenset({1, 2, 5})  # normal settlement survival
COLLAPSE_ROUNDS = frozenset({3, 4, 8})  # all settlements collapse
AGGRESSIVE_ROUNDS = frozenset({6, 7})  # massive settlement expansion

# ---------------------------------------------------------------------------
# Dynamic cell classification
# ---------------------------------------------------------------------------

DYNAMIC_SETTLEMENT_RADIUS = 3  # distance threshold for dynamic cells near settlements
DYNAMIC_FOREST_RADIUS = 3  # forest cells within this distance of settlements are dynamic
DYNAMIC_DIST_THRESHOLD = 3  # backtest: distance threshold for dynamic cell identification

# ---------------------------------------------------------------------------
# Feature predictor
# ---------------------------------------------------------------------------

SETTLEMENT_DENSITY_WINDOW = 7  # window size for settlement density filter
SETTLEMENT_DENSITY_MAX_BIN = 5  # density values >= this are capped
FEATURE_PROB_FLOOR = 0.01  # minimum probability per class after lookup

# ---------------------------------------------------------------------------
# Regime detection thresholds
# ---------------------------------------------------------------------------

REGIME_COLLAPSE_THRESHOLD = 0.05  # settlement survival rate below this = collapse
REGIME_AGGRESSIVE_THRESHOLD = 0.35  # settlement survival rate above this = aggressive
QUERY_DELAY_SECONDS = 0.2  # delay between API queries to avoid rate limiting
