"""
Concept vocabulary for probing egocentric vs allocentric understanding.

This vocabulary is designed to test whether contrastive learning shifts
visual representations from egocentric (self-focused) to allocentric
(team/enemy/global-focused) understanding in CS:GO gameplay.

Categories:
- Egocentric: Self-centered perception, actions, and state
- Allocentric: Teammate awareness, enemy awareness
- Global: Game state, round outcomes, strategic situations
- Spatial: Environmental and location context
"""

# =============================================================================
# EGOCENTRIC CONCEPTS (Self-focused understanding)
# =============================================================================

EGOCENTRIC_VISUAL = [
    # First-person perspective
    "first person view",
    "first person shooter perspective",
    "looking through my eyes",
    "my point of view",
    "what I see in front of me",
    "my field of vision",
    "my line of sight",
    # Weapon and hands
    "my hands holding a weapon",
    "my gun in frame",
    "holding a rifle",
    "holding a pistol",
    "holding a knife",
    "reloading my weapon",
    "switching weapons",
    "my crosshair on screen",
    "aiming down sights",
    "looking through scope",
    # HUD elements (self-focused)
    "my health bar",
    "my ammo count",
    "my money display",
    "my equipment slots",
]

EGOCENTRIC_MOVEMENT = [
    # Movement states
    "I am running",
    "I am walking slowly",
    "I am sprinting",
    "I am crouching",
    "I am jumping",
    "I am falling",
    "I am climbing",
    "I am standing still",
    "I am moving forward",
    "I am moving backward",
    "I am strafing left",
    "I am strafing right",
    # Speed and momentum
    "moving at high speed",
    "moving cautiously",
    "stopped completely",
    "quick movement",
    "slow careful movement",
]

EGOCENTRIC_ACTIONS = [
    # Combat actions
    "I am shooting",
    "I am about to fire",
    "I am taking aim",
    "I am throwing a grenade",
    "I am planting the bomb",
    "I am defusing the bomb",
    # Tactical actions
    "I am peeking around corner",
    "I am checking angles",
    "I am clearing a room",
    "I am holding an angle",
    "I am watching a doorway",
    "I am camping a position",
    # Utility usage
    "I am throwing a flashbang",
    "I am throwing smoke",
    "I am throwing molotov",
    "I am using utility",
]

EGOCENTRIC_STATE = [
    # Danger assessment (self)
    "I am in danger",
    "I am safe",
    "I am exposed",
    "I am hidden",
    "I am vulnerable",
    "I am protected",
    "I am taking damage",
    "I am low on health",
    "I am full health",
    # Situational awareness (self)
    "I am alone",
    "I am surrounded",
    "I am flanking",
    "I am being flanked",
    "I am in a good position",
    "I am in a bad position",
    "I am trapped",
    "I have an escape route",
]

# =============================================================================
# ALLOCENTRIC CONCEPTS - TEAMMATE (Team-focused understanding)
# =============================================================================

TEAMMATE_PRESENCE = [
    # Teammate visibility
    "teammate visible",
    "teammates nearby",
    "teammate in frame",
    "ally player on screen",
    "friendly player visible",
    "team member present",
    "multiple teammates visible",
    "teammate in the distance",
    "teammate close by",
    "teammate behind me",
    # Teammate count
    "one teammate visible",
    "two teammates visible",
    "three teammates visible",
    "four teammates visible",
    "full team present",
    "teammates missing",
    "teammate alone",
]

TEAMMATE_ACTIONS = [
    # Movement
    "teammate is running",
    "teammate is crouching",
    "teammate is holding position",
    "teammate is advancing",
    "teammate is retreating",
    "teammate is rotating",
    # Combat
    "teammate is shooting",
    "teammate is in a firefight",
    "teammate is engaging enemy",
    "teammate is taking fire",
    "teammate got a kill",
    "teammate died",
    "teammate is low health",
    # Support
    "teammate is covering me",
    "teammate is watching flank",
    "teammate is throwing utility",
    "teammate is supporting",
]

TEAMMATE_COORDINATION = [
    # Team tactics
    "teammates working together",
    "coordinated team push",
    "team executing strategy",
    "teammates stacking a site",
    "team split across map",
    "teammates regrouping",
    "team formation",
    "trading kills with teammate",
    # Positioning
    "teammates covering angles",
    "teammates crossfire setup",
    "teammate holding bombsite",
    "teammates on same site",
    "teammates spread out",
    "teammate in position",
    "teammate out of position",
]

TEAMMATE_STATE = [
    # Team status
    "teammates are alive",
    "teammate just died",
    "teammate is last alive",
    "teammates are winning",
    "teammates are losing",
    "team has numbers advantage",
    "team is outnumbered",
    "team morale is high",
    "team is struggling",
]

# =============================================================================
# ALLOCENTRIC CONCEPTS - ENEMY (Opponent-focused understanding)
# =============================================================================

ENEMY_PRESENCE = [
    # Enemy visibility
    "enemy visible",
    "enemy player spotted",
    "hostile in sight",
    "opponent on screen",
    "enemy in frame",
    "enemy in the distance",
    "enemy up close",
    "enemy behind cover",
    "enemy exposed",
    # Enemy count
    "one enemy visible",
    "two enemies visible",
    "multiple enemies",
    "enemy team present",
    "enemies everywhere",
    "no enemies visible",
    "enemy hiding",
]

ENEMY_ACTIONS = [
    # Movement
    "enemy is running",
    "enemy is pushing",
    "enemy is retreating",
    "enemy is rotating",
    "enemy is flanking",
    "enemy is rushing",
    "enemy is sneaking",
    # Combat
    "enemy is shooting at me",
    "enemy is aiming at me",
    "enemy is throwing utility",
    "enemy is planting bomb",
    "enemy is defusing bomb",
    "enemy got a kill",
    "enemy died",
]

ENEMY_THREAT = [
    # Threat assessment
    "enemy threat nearby",
    "dangerous enemy position",
    "enemy has advantage",
    "enemy is vulnerable",
    "enemy is dangerous",
    "enemy is weak",
    "enemy is strong",
    "enemy in power position",
    "enemy caught off guard",
    # Tactical situation
    "enemy is camping",
    "enemy is holding angle",
    "enemy is peeking",
    "enemy is baiting",
    "enemy is lurking",
    "enemy aggression",
    "enemy playing passive",
]

# =============================================================================
# GLOBAL GAME STATE CONCEPTS
# =============================================================================

BOMB_STATE = [
    # Bomb status
    "bomb is planted",
    "bomb not planted yet",
    "bomb is being planted",
    "bomb is being defused",
    "bomb is about to explode",
    "bomb timer running",
    "bomb at A site",
    "bomb at B site",
    "bomb dropped on ground",
    "carrying the bomb",
]

ROUND_STATE = [
    # Round progress
    "round just started",
    "early round",
    "mid round",
    "late round",
    "round is almost over",
    "round ending soon",
    "overtime round",
    "match point",
    # Round outcome
    "round is being won",
    "round is being lost",
    "close round",
    "one-sided round",
    "comeback happening",
    "clutch situation",
]

COMBAT_STATE = [
    # Combat intensity
    "active combat",
    "firefight in progress",
    "intense gunfight",
    "shots being fired",
    "explosions happening",
    "chaotic battle",
    "quiet moment",
    "calm before storm",
    "lull in action",
    # Combat outcome
    "kills happening",
    "players dying",
    "trade kills",
    "multi-kill",
    "ace in progress",
]

ECONOMY_STATE = [
    # Economic situation
    "eco round",
    "full buy round",
    "force buy round",
    "saving money",
    "team is broke",
    "team has money",
    "expensive weapons visible",
    "pistol round",
]

# =============================================================================
# SPATIAL AND ENVIRONMENTAL CONCEPTS
# =============================================================================

MAP_LOCATIONS = [
    # General locations
    "bombsite A",
    "bombsite B",
    "middle area",
    "spawn area",
    "connector area",
    "main entrance",
    "back entrance",
    "sniper nest",
    "heaven position",
    "hell position",
]

ENVIRONMENT_TYPE = [
    # Indoor vs outdoor
    "indoor area",
    "outdoor area",
    "enclosed space",
    "open space",
    "corridor or hallway",
    "large room",
    "small room",
    "narrow passage",
    "wide area",
    # Structures
    "near a wall",
    "behind cover",
    "near a box",
    "on stairs",
    "near a door",
    "near a window",
    "under an arch",
    "in a corner",
]

TACTICAL_POSITIONS = [
    # Position quality
    "defensive position",
    "offensive position",
    "power position",
    "weak position",
    "exposed position",
    "safe position",
    "chokepoint",
    "crossfire position",
    # Cover
    "good cover available",
    "no cover nearby",
    "partial cover",
    "head glitch spot",
    "one-way angle",
]

VISIBILITY_CONDITIONS = [
    # Lighting
    "well lit area",
    "dark area",
    "shadowy corner",
    "bright outdoor",
    "dim indoor",
    # Obstructions
    "smoke blocking view",
    "flash effect",
    "clear visibility",
    "obstructed view",
    "long sightline",
    "short sightline",
]

# =============================================================================
# AGGREGATE CATEGORIES
# =============================================================================

CONCEPT_CATEGORIES = {
    # Egocentric (self-focused)
    "egocentric_visual": EGOCENTRIC_VISUAL,
    "egocentric_movement": EGOCENTRIC_MOVEMENT,
    "egocentric_actions": EGOCENTRIC_ACTIONS,
    "egocentric_state": EGOCENTRIC_STATE,
    # Allocentric - Teammate
    "teammate_presence": TEAMMATE_PRESENCE,
    "teammate_actions": TEAMMATE_ACTIONS,
    "teammate_coordination": TEAMMATE_COORDINATION,
    "teammate_state": TEAMMATE_STATE,
    # Allocentric - Enemy
    "enemy_presence": ENEMY_PRESENCE,
    "enemy_actions": ENEMY_ACTIONS,
    "enemy_threat": ENEMY_THREAT,
    # Global game state
    "bomb_state": BOMB_STATE,
    "round_state": ROUND_STATE,
    "combat_state": COMBAT_STATE,
    "economy_state": ECONOMY_STATE,
    # Spatial
    "map_locations": MAP_LOCATIONS,
    "environment_type": ENVIRONMENT_TYPE,
    "tactical_positions": TACTICAL_POSITIONS,
    "visibility_conditions": VISIBILITY_CONDITIONS,
}

# High-level category groupings for visualization
CATEGORY_GROUPS = {
    "egocentric": ["egocentric_visual", "egocentric_movement", "egocentric_actions", "egocentric_state"],
    "teammate": ["teammate_presence", "teammate_actions", "teammate_coordination", "teammate_state"],
    "enemy": ["enemy_presence", "enemy_actions", "enemy_threat"],
    "global": ["bomb_state", "round_state", "combat_state", "economy_state"],
    "spatial": ["map_locations", "environment_type", "tactical_positions", "visibility_conditions"],
}

# Color scheme for visualization
CATEGORY_COLORS = {
    # Egocentric - warm colors (red/orange)
    "egocentric_visual": "#e74c3c",
    "egocentric_movement": "#c0392b",
    "egocentric_actions": "#e67e22",
    "egocentric_state": "#d35400",
    # Teammate - green
    "teammate_presence": "#2ecc71",
    "teammate_actions": "#27ae60",
    "teammate_coordination": "#1abc9c",
    "teammate_state": "#16a085",
    # Enemy - blue
    "enemy_presence": "#3498db",
    "enemy_actions": "#2980b9",
    "enemy_threat": "#2c3e50",
    # Global - purple
    "bomb_state": "#9b59b6",
    "round_state": "#8e44ad",
    "combat_state": "#7d3c98",
    "economy_state": "#6c3483",
    # Spatial - gray
    "map_locations": "#95a5a6",
    "environment_type": "#7f8c8d",
    "tactical_positions": "#bdc3c7",
    "visibility_conditions": "#aab7b8",
}

# Group-level colors (simplified)
GROUP_COLORS = {
    "egocentric": "#e74c3c",   # Red
    "teammate": "#2ecc71",     # Green
    "enemy": "#3498db",        # Blue
    "global": "#9b59b6",       # Purple
    "spatial": "#95a5a6",      # Gray
}

# Flatten all concepts for easy iteration
ALL_CONCEPTS = []
CONCEPT_TO_CATEGORY = {}
CONCEPT_TO_GROUP = {}

for category, concepts in CONCEPT_CATEGORIES.items():
    for concept in concepts:
        ALL_CONCEPTS.append(concept)
        CONCEPT_TO_CATEGORY[concept] = category
        # Find which group this category belongs to
        for group, categories in CATEGORY_GROUPS.items():
            if category in categories:
                CONCEPT_TO_GROUP[concept] = group
                break


def get_concepts_by_group(group_name: str) -> list:
    """Get all concepts belonging to a high-level group."""
    concepts = []
    for category in CATEGORY_GROUPS.get(group_name, []):
        concepts.extend(CONCEPT_CATEGORIES.get(category, []))
    return concepts


def get_category_color(category: str) -> str:
    """Get color for a category."""
    return CATEGORY_COLORS.get(category, "#7f8c8d")


def get_group_color(group: str) -> str:
    """Get color for a high-level group."""
    return GROUP_COLORS.get(group, "#7f8c8d")


if __name__ == "__main__":
    # Print statistics
    print(f"Total concepts: {len(ALL_CONCEPTS)}")
    print("\nConcepts per category:")
    for cat, concepts in CONCEPT_CATEGORIES.items():
        print(f"  {cat}: {len(concepts)}")
    print("\nConcepts per group:")
    for group, categories in CATEGORY_GROUPS.items():
        total = sum(len(CONCEPT_CATEGORIES[c]) for c in categories)
        print(f"  {group}: {total}")
