"""Every tunable constant in the pipeline, in one place.

Values here are physics-informed defaults, chosen from what is physically possible on a football
pitch and from self-supervised checks on the tracking itself (see layer0_clean/evaluate.py).
None of them is tuned against any ground truth.
"""

# --- time -------------------------------------------------------------------------------------
FPS = 25
DT = 1.0 / FPS

# --- layer 0: ball cleaning -------------------------------------------------------------------
# Fastest plausible ball speed. Professional shots reach ~35 m/s; 42 leaves room for noise.
BALL_V_MAX = 42.0                 # m/s
# Position jitter allowed on top of BALL_V_MAX when testing whether two samples can be the same
# ball. Automatic tracking is noisy at the decimetre level.
BALL_JITTER = 0.6                 # m
# A run of consecutive ball detections shorter than this, that is not consistent with the ball
# on either side of it, is treated as a false detection.
BALL_MIN_TRACKLET = 3             # frames
# Longest gap that is filled by interpolation. Longer gaps stay missing (layer 2 handles them).
# Set from the gap-fill benchmark: hiding real samples and refilling them with a cubic Hermite
# curve gives a p90 worst-point error of ~0.3 m at 25 frames (linear: ~2.2 m).
BALL_MAX_FILL_GAP = 25            # frames (1.0 s)
BALL_FILL_METHOD = "hermite"
# Savitzky-Golay smoothing of the ball within a continuous run.
BALL_SG_WINDOW = 5                # frames, odd
BALL_SG_ORDER = 2

# --- layer 0: players -------------------------------------------------------------------------
# Fastest plausible player speed (sprinters top out around 10-11 m/s). A larger jump between
# consecutive frames of one track slot is an identity switch, and the slot's run is split there.
PLAYER_V_MAX = 14.0               # m/s
PLAYER_JITTER = 0.3               # m
PLAYER_SG_WINDOW = 9              # frames, odd
PLAYER_SG_ORDER = 2
# Two slots of the same team closer than this on the same frame are one person tracked twice.
DUPLICATE_RADIUS = 0.6            # m

# --- layer 0: team repair (cross-team identity switches) --------------------------------------
# An opponent's track ending this close to a slot is a plausible moment for the tracker to have
# swapped the two people.
SWITCH_RADIUS = 2.0               # m
TEAM_REPAIR_MIN_EPISODE = 5       # frames of 12-vs-10 before a repair is attempted
TEAM_REPAIR_MIN_GAIN = 10         # excess player-frames a repair must remove
TEAM_REPAIR_PASSES = 4
# A slot whose run began at most this long before a 12-vs-10 episode may be relabelled whole.
TEAM_REPAIR_LEAD_IN = 125         # frames (5 s)

# --- layer 1: features ------------------------------------------------------------------------
PRESSURE_RADII = (2.0, 5.0)       # m, players of a team within these of the ball
CONTROL_RADIUS = 1.5              # m, a player this close can play the ball
BALL_STILL_SPEED = 0.5            # m/s, slower than this the ball is at rest

# --- layer 2: touches -------------------------------------------------------------------------
# The provider's ball track is smoothed on a ~0.5 s scale, so a touch is not a sharp jump but a
# local minimum of ball speed (reception, or the moment before a kick) or a sharp turn.
TOUCH_SPEED_MIN_PROMINENCE = 2.0  # m/s the speed must dip by, either side, to count
TOUCH_TURN_K = 6                  # frames each side for the heading comparison
TOUCH_TURN_DEG = 30.0             # heading change that counts as a touch
TOUCH_TURN_MIN_SPEED = 2.0        # m/s; heading is meaningless for a ball at rest
# A free ball redirected by a player turns moderately; a near-reversal at speed is the tracker
# jumping, not a touch (it is no closer to players than a random frame).
TOUCH_TURN_MAX_DEG = 90.0
TOUCH_TURN_SPEED_RANGE = (4.0, 14.0)   # m/s where a turn is a credible redirect
TOUCH_SPEED_MIN_MAX_PROMINENCE = 8.0  # m/s; a bigger dip is the ball stopping dead, not a touch
TOUCH_MERGE = 6                   # frames; candidates closer than this are one touch
TOUCH_ATTRIB_WINDOW = 5           # frames each side to find the toucher
TOUCH_ATTRIB_RADIUS = 2.5         # m; no player this close -> unattributed
# "ball with a player": nearest player close and moving with the ball
CONTROL_D = 1.5                   # m
CONTROL_RELV = 3.0                # m/s
CONTROL_MIN_FRAMES = 8            # 0.32 s
CONTROL_MAX_BREAK = 4             # frames of interruption tolerated inside one segment

# --- layer 3: dead ball -----------------------------------------------------------------------
# Evidence per frame, in log-odds (nats) of dead vs alive. Physical defaults; see
# layer3_dead_ball/README.md for how each was checked.
DEAD_OUTSIDE_M = 0.5              # ball this far beyond a line is out of play
DEAD_W_OUTSIDE = 4.0
DEAD_W_GAP_LONG = 1.5             # ball missing for >= DEAD_GAP_LONG
DEAD_GAP_LONG = 125               # frames (5 s)
DEAD_W_GAP_MID = 0.5              # ball missing for 2-5 s
DEAD_GAP_MID = 50
DEAD_W_GAP_EDGE = 1.0             # ...and it was last seen at / beyond a line
DEAD_EDGE_M = 2.0
DEAD_SPEED_PIVOT = 1.6            # m/s mean player speed where the evidence is neutral
DEAD_W_SPEED = 1.5                # per m/s below the pivot (3 s mean)
DEAD_W_BALL_FAST = -1.0           # ball moving fast inside the pitch: play is on
DEAD_BALL_FAST = 8.0              # m/s
DEAD_W_CONTROL = -1.0             # a player controlling the ball away from the lines
DEAD_W_STILL = 1.5                # ball at rest for >= DEAD_STILL_FRAMES
DEAD_STILL_SPEED = 1.0            # m/s
DEAD_STILL_FRAMES = 37            # 1.5 s
DEAD_W_SPOT = 1.0                 # ...at a restart spot
DEAD_SPOT_M = 2.0
DEAD_W_SET_PIECE = 1.0            # inside a Smart Tagging set-piece window, before its middle
DEAD_W_KICKOFF_SHAPE = 3.0        # both teams wholly in their own halves, ball at the centre
DEAD_MIN_DEAD_S = 3.0             # a stoppage is never shorter than this
DEAD_MIN_ALIVE_S = 2.0            # nor is play between two stoppages
DEAD_SWITCH_COST = 8.0            # nats to switch alive <-> dead (suppresses flicker)
DEAD_SNAP_WINDOW = 75             # frames each side to snap a boundary to a kick / line crossing

# --- layer 4: possession model ----------------------------------------------------------------
# Certain possession = a player controlling the ball, uncontested, for at least this long.
POSS_LABEL_MIN_CONTROL = 25       # frames (1 s)
POSS_LABEL_MAX_CONTEST = 0.25
# Also label the ball travelling between two controls of the same team (a completed pass)
# when the hand-over takes at most this long.
POSS_LABEL_PASS_GAP = 75          # frames (3 s)
POSS_FOLDS = 10                   # contiguous time blocks for out-of-fold training
POSS_BUFFER = 250                 # frames dropped from training either side of a test block

# --- layer 5: sequence decoder ----------------------------------------------------------------
# Per-frame log-odds that home has the ball, summed over each candidate segment:
SEQ_W_CONTROL = 1.0               # nats per frame of a player controlling the ball (x (1 - contest))
# Chosen on the bench (possession/bench, spellF1 on three other games, models never trained on
# the game they are scored on). Before the bench: W_PLAYERS 0.02, W_FULL 0, no change points.
SEQ_W_PLAYERS = 0.0               # x logit(p_home_players) per frame (self-trained, players only)
SEQ_W_FULL = 0.35                 # x logit(p_home_full) per frame (cross-game, players + ball)
# Cost of a possession change at a candidate event, and how much a reception by the new team
# (or a close touch) pays back:
SEQ_SWITCH_COST = 4.0             # nats
SEQ_W_RECEIVE = 8.0               # nats, uncontested reception of >= 1 s by the new team
SEQ_W_TOUCH = 1.0                 # nats, touch within 1 m by the new team
SEQ_W_KICK = 12.0                  # nats, a kick by the new team (first-time pass, clearance)
# Possession-length prior (log-normal shape), in nats at one sigma
SEQ_DUR_MEDIAN_S = 10.0
SEQ_DUR_SIGMA = 1.0
SEQ_W_DUR = 0.3
SEQ_W_CP = 1.0                    # x log P(change to team) / P(no change), learned change points
SEQ_CP_OFFSET = 0.0               # nats added to every learned change point
SEQ_DENSE_GRID = 0                # frames: if > 0, candidate points everywhere on this grid
SEQ_GAP_GRID = 25                 # frames: candidate points every second inside ball gaps > 2 s
# kicks: the ball accelerating away from a player who touched it only briefly (a first-time pass,
# a clearance). The team that last played the ball keeps it until the other team controls it.
KICK_RISE = 5.0                   # m/s the ball's speed must rise by ...
KICK_WINDOW = 12                  # ... within this many frames
KICK_RADIUS = 1.5                 # m, kicker within this of the ball at the start of the rise
