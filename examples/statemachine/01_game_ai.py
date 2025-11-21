"""
Example: Game AI State Machine

Demonstrates finite state machine for game character AI.
Models an enemy guard with patrol, chase, and attack states.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.statemachine import (
    StateMachineOperations, State, Transition,
    TransitionType
)


def main():
    """Create and simulate a game AI state machine"""

    print("=== Game AI State Machine ===\n")

    # Callbacks for state actions
    def on_enter_patrol(context):
        print(f"  → Entering PATROL state")
        context['alert_level'] = 0

    def on_update_patrol(context, dt):
        context['patrol_time'] = context.get('patrol_time', 0) + dt
        if context['patrol_time'] > 2.0:
            print(f"    [Patrol] Continuing patrol... (time: {context['patrol_time']:.1f}s)")
            context['patrol_time'] = 0

    def on_enter_alert(context):
        print(f"  → Entering ALERT state (investigating)")
        context['alert_level'] = 1

    def on_enter_chase(context):
        print(f"  → Entering CHASE state (enemy spotted!)")
        context['chase_time'] = 0
        context['alert_level'] = 2

    def on_update_chase(context, dt):
        context['chase_time'] = context.get('chase_time', 0) + dt
        distance = context.get('player_distance', 100)
        print(f"    [Chase] Chasing player (distance: {distance:.1f}m, time: {context['chase_time']:.1f}s)")

    def on_enter_attack(context):
        print(f"  → Entering ATTACK state (engaging!)")

    def on_update_attack(context, dt):
        print(f"    [Attack] Attacking player!")

    def on_exit_attack(context):
        print(f"  ← Exiting ATTACK state")

    # Create states
    patrol = State("patrol", on_enter=on_enter_patrol, on_update=on_update_patrol)
    alert = State("alert", on_enter=on_enter_alert)
    chase = State("chase", on_enter=on_enter_chase, on_update=on_update_chase)
    attack = State("attack", on_enter=on_enter_attack, on_update=on_update_attack,
                  on_exit=on_exit_attack)

    # Create state machine
    fsm = StateMachineOperations.create("GuardAI", "patrol")
    fsm = StateMachineOperations.add_state(fsm, patrol)
    fsm = StateMachineOperations.add_state(fsm, alert)
    fsm = StateMachineOperations.add_state(fsm, chase)
    fsm = StateMachineOperations.add_state(fsm, attack)

    # Define transitions with guards
    def player_heard(ctx):
        return ctx.get('noise_level', 0) > 0.5

    def player_seen(ctx):
        return ctx.get('player_distance', 100) < 20

    def player_close(ctx):
        return ctx.get('player_distance', 100) < 5

    def player_escaped(ctx):
        return ctx.get('player_distance', 100) > 30

    def chase_timeout(ctx):
        return ctx.get('chase_time', 0) > 10.0

    # Patrol → Alert (on noise)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "patrol", "alert", event="noise_detected"
    ))

    # Alert → Chase (on sight)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "alert", "chase", guard=player_seen,
        transition_type=TransitionType.AUTOMATIC
    ))

    # Alert → Patrol (timeout)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "alert", "patrol",
        transition_type=TransitionType.TIMEOUT,
        timeout=5.0
    ))

    # Chase → Attack (player close)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "chase", "attack", guard=player_close,
        transition_type=TransitionType.AUTOMATIC
    ))

    # Chase → Patrol (player escaped)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "chase", "patrol", guard=player_escaped,
        transition_type=TransitionType.AUTOMATIC
    ))

    # Chase → Patrol (timeout)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "chase", "patrol", guard=chase_timeout,
        transition_type=TransitionType.AUTOMATIC
    ))

    # Attack → Chase (player retreats)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "attack", "chase", event="player_retreats"
    ))

    # Attack → Patrol (player defeated)
    fsm = StateMachineOperations.add_transition(fsm, Transition(
        "attack", "patrol", event="player_defeated"
    ))

    # Start the state machine
    fsm = StateMachineOperations.start(fsm)

    # Simulate game scenario
    print("=== Simulation Start ===\n")

    # Patrol for a while
    print("Time 0-2s: Guard patrolling...")
    for i in range(4):
        fsm = StateMachineOperations.update(fsm, 0.5)

    # Guard hears noise
    print("\nTime 2s: Guard hears suspicious noise!")
    fsm.context['noise_level'] = 0.8
    fsm = StateMachineOperations.send_event(fsm, "noise_detected")

    # Guard investigates
    print("\nTime 2-4s: Guard investigating...")
    for i in range(4):
        fsm = StateMachineOperations.update(fsm, 0.5)

    # Player spotted!
    print("\nTime 4s: Player spotted nearby!")
    fsm.context['player_distance'] = 15.0
    fsm = StateMachineOperations.update(fsm, 0.1)

    # Chase sequence
    print("\nTime 4-6s: Guard chasing player...")
    for i in range(4):
        fsm.context['player_distance'] = max(3.0, 15.0 - i * 2.0)
        fsm = StateMachineOperations.update(fsm, 0.5)

    # Player very close - attack!
    print("\nTime 6s: Player in attack range!")
    fsm.context['player_distance'] = 2.0
    fsm = StateMachineOperations.update(fsm, 0.1)

    # Attack
    print("\nTime 6-7s: Combat!")
    for i in range(2):
        fsm = StateMachineOperations.update(fsm, 0.5)

    # Player defeated
    print("\nTime 7s: Player defeated!")
    fsm = StateMachineOperations.send_event(fsm, "player_defeated")

    print("\n=== Simulation End ===")
    print(f"\nFinal state: {fsm.current_state}")

    # Print state machine diagram
    print("\n=== State Machine Diagram (Graphviz DOT format) ===")
    print(StateMachineOperations.to_graphviz(fsm))


if __name__ == "__main__":
    main()
