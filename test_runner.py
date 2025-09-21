import torch
import inference
from phase_tokenizer import PhaseTokenizer

# Ensure tokenizer and a_temp are initialized the same way as interactive loop
if inference.tokenizer is None:
    inference.tokenizer = PhaseTokenizer(dim=128, h=0.05, i=1.0)
if inference.a_temp is None:
    inference.a_temp = torch.full((1, 1), 0.5, dtype=torch.float32)

responder = inference.build_chat_responder(max_examples=500)

inputs = ["hi", "who are you?"]
for ui in inputs:
    print(f"\n=== INPUT: {ui}")
    phase_input = inference.tokenizer.encode_text(ui)
    dummy_embed = phase_input
    dummy_label = torch.tensor([0])
    with torch.no_grad():
        _, input_phase = inference.model(dummy_embed, inference.a_temp, dummy_label, dummy_label, epoch=999)

    for use_norm in (False, True):
        print(f"\n--- use_l2_norm={use_norm} ---")
        try:
            reply = inference.compose_reply_from_responder(input_phase.squeeze(0), responder, k=5, temperature=0.6, out_len=100, use_l2_norm=use_norm)
            resp_text = reply['composed_text']
            resp_text = inference.make_direct_reply(ui, resp_text)
            print("REPLY:", resp_text)
        except Exception as e:
            print("ERROR during compose (use_l2_norm=", use_norm, "):", e)

print('\n[TestRunner] done')
