# v9 preliminary changes

1. Existing v8 already contained:
- NodeReport
- active_node_history
- validity_history_by_node

2. Added:
- cooperative/communication.py
- examples/run_multi_sat_node_dropout.py

3. Communication layer currently supports deterministic dropout.
Next extensions:
- packet loss
- delay buffer
- asynchronous fusion
