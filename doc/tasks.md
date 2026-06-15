# Downstream Tasks

The Stage 2 task registry: what tasks exist, how they're defined, and how a `task_id` becomes a fully-configured run.

Relevant code: `src/scripts/task_creator/task_definitions.py`, `src/scripts/task_creator/create_all_labels.py`, `src/utils/config_utils.py`.

## How a task is selected

You pick a task with `task.task_id` on the CLI:

```bash
python main.py --mode train --task downstream task.task_id=enemy_location_5s
```

`apply_task_config` (`config_utils.py`) then:

1. finds the nearest `task_definitions.csv` (searches `<data>/<map>/labels/`, then `<data>/labels/`, then a glob),
2. looks up the row for `task_id` and fills `task.ml_form`, `task.num_classes`, `task.output_dim`,
3. sets `data.labels_filename = <task_id>.csv` (under `labels/all_tasks`),
4. derives `task.label_column` (single `label`, or `;`-separated `label_0;label_1;…` for multi-output tasks),
5. for location/place tasks, overrides `output_dim` to the number of named places on the map.

If `task_definitions.csv` can't be found, it falls back to the YAML values.

> `task_definitions.csv` lives under `DATA_BASE_PATH` (generated, not committed). The authoritative in-repo enumeration of tasks is `TASK_CONFIGS` in `create_all_labels.py`.

## ML forms

Each task has an `ml_form` that determines head shape, loss, and metrics (see [architecture.md](architecture.md#heads-by-ml_form)):

- `binary_cls`, `multi_cls`, `multi_label_cls`, `regression`.

Tasks also carry a `TemporalType` — `nowcast` (predict the *current* state) or `forecast` (predict a future state at horizon `Ns`).

## Task registry

The ~39 implemented tasks (from `TASK_CONFIGS`), grouped by category. The `Ns` suffix is the forecast horizon in seconds (`0s` = nowcast).

### Location (`location`)
Predict which named map region an agent occupies (output dim = map's place count).
- Self: `self_location_0s`, `self_location_5s`, `self_location_10s`, `self_location_20s`
- Teammate: `teammate_location_0s`, `teammate_location_5s`, `teammate_location_10s`, `teammate_location_20s`
- Enemy: `enemy_location_0s`, `enemy_location_5s`, `enemy_location_10s`, `enemy_location_20s`

### Coordination (`coordination`)
- `teammate_aliveCount`, `enemy_aliveCount` — how many are alive
- `teammate_movementDir` — teammates' movement directions
- `teammate_speed` — teammates' speed (regression)

### Combat (`combat`)
- `global_anyKill_{0s,5s,10s,20s}` — any kill imminent
- `self_death_{5s,10s,20s}` — POV player dies soon
- `self_kill_{0s,5s,10s,20s}` — POV player gets a kill soon
- `teammate_inCombat`, `self_inCombat`

### Bomb (`bomb`)
- `global_bombPlanted`, `global_bombSite`, `global_willPlant`, `global_postPlantOutcome`
- `global_roundWinner`, `global_roundOutcome` (multi-class reason)

### Spatial (`spatial`)
- `self_movementDir` — POV movement direction
- `self_speed` — POV speed (regression)

### Not yet implemented
`NOT_IMPLEMENTED_TASKS` in `create_all_labels.py`: `headshot_next_kill`, `bomb_carrier_dist`, `pov_place_cls`, `area_control_mid`, `imminent_shot_self_2s`, `team_executing`, `team_rotating`.

## Map vocabularies

`task_definitions.py` defines the categorical vocabularies used to size outputs and decode predictions:

- **Places**: `MIRAGE_PLACES` (23), `DUST2_PLACES` (24), `INFERNO_PLACES` (24). `get_place_names_for_map` normalizes the map name (strips a `de_` prefix, lowercases, defaults to Mirage).
- **Movement**: 8 compass directions + `STATIONARY` (`NUM_DIRECTIONS = 9`).
- **Round outcomes**: `t_killed`, `ct_killed`, `bomb_exploded`, `bomb_defused`, `time_ran_out` (`NUM_OUTCOMES = 5`).
- **Weapons**: `WEAPONS` / `WEAPON_TO_IDX`.

Default maps used by the generators: `dust2`, `inferno`, `mirage`.

## Generating the label CSVs

Label CSVs are produced by `create_all_labels.py` (see [data-preparation.md](data-preparation.md)). Each implemented task in `TASK_CONFIGS` maps to a creator class in `task_creator/task_creator_helper/` that turns parsed trajectories/events into a `<task_id>.csv` with a generic `label` (or `label_i`) column plus the metadata columns the dataset needs.
